import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
import shutil 
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve, average_precision_score, f1_score
from .util import *
from collections import defaultdict
import math
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class WordPredictor(nn.Module):
    def __init__(self, input_dim, latent_dim):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU(x)
        x = self.fc2(x)
        x = nn.Sigmoid(x)
        return x

def find_top_words(corpus_size, k, word_freq, val_corpus, min_freq = 100):
    word_idf = {}
    for word in word_freq:
        if word_freq[word] >= min_freq and word != "<unk>" and len(word) > 1 and word in val_corpus:
            word_idf[word] = math.log(corpus_size/word_freq[word])

    s = pd.Series(data = word_idf)
    s = s.sort_values(ascending = False)
    s = s.head(k)
    return s.index.tolist()
             
    

def predict_word_presence(audio_model, train_loader, val_loaders, train_ctm, val_ctm, epochs, args, exp_dir):
    print("Beginning probe training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Found %d GPUs' % torch.cuda.device_count())

    torch.set_grad_enabled(True)

    audio_model = audio_model.to(device)

    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in audio_model.parameters() if p.requires_grad)
    print("Number of parameters: %d" % num_params)

    num_models = args.num_probes
       
    pos_weight = [3566]
    pos_weight = torch.from_numpy(np.array(pos_weight)).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight).to(device)
    optimizer = optim.Adam(audio_trainables, args.lr, weight_decay=args.weight_decay)
    
    words = {}
    word_freq = defaultdict(int)
    word_times = set()
    corpus_size = 0
    sentence = ""
    prev = ""
    with open(train_ctm, 'r') as f:
        for timestamp in f:
            components = timestamp.split(" ")
            uttid = components[0]
            if uttid != prev:
                word_times = set()
                sentence = ""
                corpus_size += 1
            prev = uttid
            word = components[4].lower()

            word_freq[word] += 1
            sentence += word + " "
            word_times.add(word)
            words[uttid] = word_times

    val_words = {}
    val_word_times = set()
    val_corpus = set()
    sentence = ""
    prev = ""
    with open(val_ctm, 'r') as f:
        for timestamp in f:
            components = timestamp.split(" ")
            uttid = components[0]
            if uttid != prev:
                val_word_times = set()
                sentence = ""
            prev = uttid
            start_time = float(components[2])
            end_time = start_time + float(components[3])
            word = components[4].lower()

            val_corpus.add(word)
            sentence += word + " "
            val_word_times.add(word)
            val_words[uttid] = val_word_times

    print("Finished reading ctm files")

    top_words = find_top_words(corpus_size, num_models, word_freq, val_corpus)

    epoch = 1
    best_precision = 0
    best_epoch = 0
    global_step = 0
    print("Finished finding top words")
    print(top_words)
    
    loss_meter = AverageMeter()
    batch_timer = AverageMeter()
  
    while epoch <= args.n_epochs:
        torch.cuda.empty_cache()
        cur_lr = adjust_learning_rate(args.lr, args.lr_decay, args.lr_decay_multiplier, optimizer, epoch)
        audio_model.train()
        epoch_time = time.time()
        batch_time = time.time()


        probe_acc = np.zeros((4, args.num_probes))
        for i, (image_input, audio_input, nframes, strings, id) in enumerate(train_loader):
            audio_input = audio_input.to(device)
            optimizer.zero_grad()
            B = audio_input.size(0)

            labels = np.zeros((len(id), 4, num_models), dtype = np.float32)
            output = audio_model(audio_input, nframes, None, probe = True)
            
            for j in range(4):
                for k, word in enumerate(top_words):
                    for l in range(len(id)):
                        try:
                            labels[l][j][k] = 1 if word in words[id[l]] else 0
                        except KeyError:
                            labels[l][j][k] = 0
            labels = torch.from_numpy(labels)

            labels = labels.reshape((len(id), 4 * num_models))
            output = output.reshape((len(id), 4 * num_models))
            
            labels = labels.to(device)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), B)
            batch_timer.update(time.time() - batch_time)

            if global_step % args.n_print_steps == 0:
                print('Epoch: [{0}][{1}/{2}]'
                          '  Time={bt.val:.3f} ({bt.avg:.3f})'
                          '  Loss total={loss.val:.4f} ({loss.avg:.4f})'.format(
                           epoch, i, len(train_loader), bt=batch_timer, loss=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            batch_time = time.time()
            global_step += 1

            del (loss, output)

        avg_probe_acc = np.zeros((4, args.num_probes))
        avg_positives = 0
        avg_metrics = np.zeros((3,4))
        for k in range(len(val_loaders)):
            acc, positives, metrics = validate(audio_model, val_loaders[k], args, val_words, top_words)
            avg_probe_acc += acc
            avg_metrics += metrics
            avg_positives += positives
        
        avg_metrics /= len(val_loaders)
        avg_probe_acc /= len(val_loaders)
        avg_positives /= len(val_loaders)

        avg_acc = np.mean(avg_probe_acc)
        avg_layer_acc = np.mean(avg_probe_acc, axis = 1).tolist()

        max_avg_precision = np.amax(avg_metrics[1])

        avg_metrics = avg_metrics.tolist()

        print("Finished epoch %d. Acc: %f. Positives: %f" % (epoch, avg_acc, avg_positives))
        print("Average acc")
        print(avg_layer_acc)
        print("Mean eers")
        print(avg_metrics[0])
        print("Mean average precisions")
        print(avg_metrics[1])
        print("Mean F1s")
        print(avg_metrics[2])

        torch.save(audio_model.state_dict(),
                "%s/models/audio_model.iter.pth" % (exp_dir))
        torch.save(image_model.state_dict(),
                "%s/models/image_model.iter.pth" % (exp_dir))
        torch.save(optimizer.state_dict(), 
                "%s/models/optim_state.iter.pth" % (exp_dir))

        if max_avg_precision > best_precision:
            best_precision = max_avg_precision
            best_epoch = epoch

            shutil.copyfile("%s/models/audio_model.iter.pth" % (exp_dir), 
                "%s/models/best_audio_model.pth" % (exp_dir))
            shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir), 
                "%s/models/best_image_model.pth" % (exp_dir))

        if args.save_every > 0 and epoch % args.save_every == 0:
            shutil.copyfile("%s/models/audio_model.iter.pth" % (exp_dir), 
                "%s/models/audio_model.e%d.pth" % (exp_dir, epoch))
            shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir), 
                "%s/models/image_model.e%d.pth" % (exp_dir, epoch))

        epoch += 1
    print("Finished training, best_epoch = %s, best_precision = %s" % (best_epoch, best_precision))
    return best_precision


def validate(audio_model, val_loaders, args, words, top_words):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    N_examples = len(val_loaders.dataset)

    num_positives = 0
    probe_acc = np.zeros((4, args.num_probes))
    y = []
    y_pred = []
    
    audio_model.eval()
    with torch.no_grad():
        for i, (image_input, audio_input, nframes, strings, id) in enumerate(val_loaders):
            audio_input = audio_input.to(device)
            
            output = audio_model(audio_input, nframes, None, probe = True)
            sigmoid = nn.Sigmoid()
            output = sigmoid(output)

            labels = np.zeros((len(id), 4, args.num_probes), dtype = np.int)
            for j in range(4):
                for k, word in enumerate(top_words):
                        for l in range(len(id)):
                            try:
                                labels[l][j][k] = 1 if word in words[id[l]] else 0
                                num_positives += labels[l][j][k]
                            except KeyError:
                                labels[l][j][k] = 0
                
            roc_output = output.to('cpu').numpy()
            y_pred.append(roc_output)
            y.append(labels)
    
            labels = torch.from_numpy(labels)
            labels = labels.to(device)
    
            pred = torch.where(output < 0.5, 0, 1) 

            diff = torch.sum(torch.abs(pred - labels), 0)
            diff = diff.to('cpu').numpy()
            probe_acc += diff

    y = np.concatenate(y, axis = 0)
    y_pred = np.concatenate(y_pred, axis = 0)

    eers = [[] for i in range(4)]
    avg_precisions = [[] for i in range(4)]
    f1_scores = [[] for i in range(4)]
    for i in range(4):
        for j in range(args.num_probes):
            probe_label = y[:,i,j]
            probe_pred = y_pred[:,i,j]

            if np.sum(probe_label) != 0:
                fpr, tpr, threshold = roc_curve(probe_label, probe_pred, pos_label = 1)
                eers[i].append(brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.))
                avg_precisions[i].append(average_precision_score(probe_label, probe_pred, pos_label = 1))
                f1_scores[i].append(f1_score(probe_label, np.where(probe_pred < 0.5, 0, 1), pos_label = 1))

    
    eers = np.array([np.average(np.array(eers[i])) for i in range(4)])
    avg_precisions = np.array([np.average(np.array(avg_precisions[i])) for i in range(4)])
    f1_scores = np.array([np.average(np.array(f1_scores[i])) for i in range(4)])

    return (N_examples - probe_acc) / N_examples, num_positives / (N_examples * 4 * args.num_probes), np.array([eers, avg_precisions, f1_scores], dtype = np.float32)

         

    
    
