# Author: David Harwath, Wei-Ning Hsu

from rangetree import RangeTree
import datetime
import numpy as np
import pickle
import shutil
import time
import torch
import torch.nn as nn

from models.quantizers import compute_perplexity
from .util import *


def map_skip_none(fn, it):
    """
    emulate list(map(fn, it)) but leave None as it is. 
    """
    ret = []
    for x in it:
        if x is None:
            ret.append(None)
        else:
            ret.append(fn(x))
    return ret


def numbers_to_str(nums, precision=3):
    msg = '('
    num_tmp = '%%.%df' % precision
    num_to_str = lambda x: (str(x) if x is None else num_tmp % x)
    for num in nums[:-1]:
        msg += num_to_str(num)
        msg += ', '
    msg += num_to_str(nums[-1])
    msg += ')'
    return msg


def train(audio_model, image_model, train_loader, val_loaders, args, exp_dir, resume, timestamps, splice, num_words_drop, word_dropout, unsupervised, ablate):
    # Initialize all of the statistics we want to keep track of
    batch_timer = AverageMeter()
    data_timer = AverageMeter()
    loss_timer = AverageMeter()
    bwd_timer = AverageMeter()
    loss_meter = AverageMeter()
    progress = []
    best_epoch, best_acc = 0, -np.inf
    global_step, epoch = 0, 0
    batch_time = time.time()



    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_acc, 
                time.time() - batch_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    # Create/load exp
    if resume:
        (progress, epoch, global_step, best_epoch, 
         best_acc) = load_progress("%s/progress.pkl" % exp_dir)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_acc = %.4f" % best_acc)

    # Set device and maybe load snapshot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    print('Found %d GPUs' % torch.cuda.device_count())
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    if epoch != 0:
        audio_model.load_state_dict(
                torch.load("%s/models/audio_model.iter.pth" % (exp_dir)))
        image_model.load_state_dict(
                torch.load("%s/models/image_model.iter.pth" % (exp_dir)))
        print("loaded parameters from epoch %d" % epoch)
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)

    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    if args.freeze_image_model:
        image_trainables = [p for n, p in image_model.named_parameters() \
                            if n.startswith('embedder')]
    else:
        image_trainables = [p for p in image_model.parameters() if p.requires_grad]
    trainables = audio_trainables + image_trainables
    print('Total %d trainable parameters' % len(trainables))

    if args.optim == 'sgd':
       optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    if epoch != 0:
        optimizer.load_state_dict(
                torch.load("%s/models/optim_state.iter.pth" % (exp_dir)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)
    else:
        torch.save(audio_model.state_dict(), 
                   "%s/models/audio_model.e%d.pth" % (exp_dir, epoch))
        torch.save(image_model.state_dict(), 
                   "%s/models/image_model.e%d.pth" % (exp_dir, epoch))
    epoch += 1
    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")

    # Start training
    while epoch <= args.n_epochs:
        torch.cuda.empty_cache()
        cur_lr = adjust_learning_rate(args.lr, args.lr_decay, 
                                      args.lr_decay_multiplier,
                                      optimizer, epoch)
        print('Learning rate @ %5d is %f' % (epoch, cur_lr))
        epoch_time = time.time()
        batch_time = time.time()
        audio_model.train()
        image_model.train()
        for i, (image_input, audio_input, nframes, strings, id) in enumerate(train_loader): 
            data_timer.update(time.time() - batch_time) 
            start_time = time.time()
            B = audio_input.size(0)
            T = audio_input.size(-1)
            audio_input = audio_input.to(device)
            image_input = image_input.to(device)

            #highly inefficient but can fix later
            tempMask = [[1 for j in range(len(strings))] for k in range(len(strings))]
            for j in range(len(strings)):
                for k in range(len(strings)):
                    if(strings[j] == strings[k] and k != j):
                        tempMask[j][k] = 0
            mask = torch.from_numpy(np.array(tempMask)).detach()
            mask = mask.to(device)

            optimizer.zero_grad()
            
            if word_dropout != 0:
                with torch.no_grad():
                    image_output = image_model(image_input)
                    audio_output = audio_model(audio_input, nframes, None)
                
                    max_times = compute_max_times(image_output, audio_output, num_words_drop, nframes, ablate)
                    for j in range(len(max_times)):
                        timestamp_list = max_times[j]
                        timestamp_list.sort(reverse=True)
                        for k in range(min(num_words_drop, len(timestamp_list))):
                            timestamp = timestamp_list[k]
                            if timestamp[0] != -1: 
                                word_start_time = timestamp[0]
                                word_end_time = timestamp[1]
                                
                                word_mask = np.random.choice(a=[0,1], size=None, p=[word_dropout, 1-word_dropout])
                                if word_mask == 0:
                                    radius = random.randint(24, ablate)
                                    radius //= 2
                                    word_end_time = min(2048, word_start_time + radius)
                                    word_start_time = max(0, word_start_time - radius)
                                    try:
                                        if splice: 
                                            audio_input[j] = torch.cat((audio_input[j,:,0:word_start_time], audio_input[j,:,word_end_time:2048], torch.zeros(size=(40, word_end_time-word_start_time)).to(device)), dim=1)
                                        else:
                                            audio_input[j] = torch.cat((audio_input[j,:,0:word_start_time], torch.zeros(size=(40, word_end_time-word_start_time)).to(device), audio_input[j,:,word_end_time:2048]), dim=1)

                                    except RuntimeError:
                                        continue

        
            image_output = image_model(image_input)
            audio_output = audio_model(audio_input, nframes, None)

            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nframes.floor_divide_(pooling_ratio)
            S = compute_pooldot_similarity_matrix(
                    image_output, audio_output, nframes)
            #replace w/ masked margin softmax
            #don't forget to call .detach() on the mask

            #I2A_sampled_loss = sampled_triplet_loss_from_S(S, args.margin)
            #A2I_sampled_loss = sampled_triplet_loss_from_S(S.t(), args.margin)
            #I2A_hardneg_loss = semihardneg_triplet_loss_from_S(S, args.margin)
            #A2I_hardneg_loss = semihardneg_triplet_loss_from_S(S.t(), args.margin)

            loss = masked_margin_softmax(S, args.margin, mask)


            loss_meter.update(loss.item(), B) 
            loss_timer.update(time.time() - start_time)
            start_time = time.time()

            loss.backward()
            optimizer.step()
            bwd_timer.update(time.time() - start_time)
            batch_timer.update(time.time() - batch_time)
            
            if global_step % args.n_print_steps == 0:
                print('Epoch: [{0}][{1}/{2}]'
                      '  Time={bt.val:.3f} ({bt.avg:.3f})'
                      '  Data={dt.val:.3f} ({dt.avg:.3f})'
                      '  Loss={lt.val:.3f} ({lt.avg:.3f})'
                      '  Bwd={bwdt.val:.3f} ({bwdt.avg:.3f})'
                      '  Loss total={loss.val:.4f} ({loss.avg:.4f})'.format(
                       epoch, i, len(train_loader), bt=batch_timer,
                       dt=data_timer, lt=loss_timer, bwdt=bwd_timer,
                       loss=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            batch_time = time.time()
            global_step += 1

            del (loss, S, image_output, audio_output)

        avg_acc = 0
        avg_acc_5 = 0
        avg_acc_1 = 0
        for k in range(len(val_loaders)):
            recalls = validate(audio_model, image_model, val_loaders[k], args)
            acc = (recalls['A_r10'] + recalls['I_r10']) / 2
            acc_5 = (recalls['A_r5'] + recalls['I_r5']) / 2
            acc_1 = (recalls['A_r1'] + recalls['I_r1']) / 2
            avg_acc += acc
            avg_acc_5 += acc_5
            avg_acc_1 += acc_1
        avg_acc /= len(val_loaders)
        avg_acc_5 /= len(val_loaders)
        avg_acc_1 /= len(val_loaders)

        print('Finished epoch %d. Time elapsed = %.fs. Current Time = %s, R@10 = %f, R@5 = %f, R@1 = %f' % (
              epoch, time.time() - epoch_time, datetime.datetime.now(), avg_acc, avg_acc_5, avg_acc_1))

        torch.save(audio_model.state_dict(),
                "%s/models/audio_model.iter.pth" % (exp_dir))
        torch.save(image_model.state_dict(),
                "%s/models/image_model.iter.pth" % (exp_dir))
        torch.save(optimizer.state_dict(), 
                "%s/models/optim_state.iter.pth" % (exp_dir))
        
        if avg_acc > best_acc:
            best_epoch = epoch
            best_acc = avg_acc
            shutil.copyfile("%s/models/audio_model.iter.pth" % (exp_dir), 
                "%s/models/best_audio_model.pth" % (exp_dir))
            shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir), 
                "%s/models/best_image_model.pth" % (exp_dir))

        if args.save_every > 0 and epoch % args.save_every == 0:
            shutil.copyfile("%s/models/audio_model.iter.pth" % (exp_dir), 
                "%s/models/audio_model.e%d.pth" % (exp_dir, epoch))
            shutil.copyfile("%s/models/image_model.iter.pth" % (exp_dir), 
                "%s/models/image_model.e%d.pth" % (exp_dir, epoch))

        _save_progress()
        epoch += 1

    print('Finished training. best_epoch = %s, best_acc = %s'
          % (best_epoch, best_acc))


def validate(audio_model, image_model, val_loader, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)
    
    batch_time = time.time()
    N_examples = len(val_loader.dataset)
    I_embeddings = [] 
    A_embeddings = [] 
    frame_counts = []
    image_model.eval()
    audio_model.eval()
    with torch.no_grad():
        for i, (image_input, audio_input, nframes, strings, id) in enumerate(val_loader):
            image_input = image_input.to(device)
            audio_input = audio_input.to(device)

            # compute output
            image_output = image_model(image_input)
            audio_output = audio_model(audio_input, nframes, None)

            image_output = image_output.to('cpu').detach()
            audio_output = audio_output.to('cpu').detach()

            I_embeddings.append(image_output)
            A_embeddings.append(audio_output)
            
            pooling_ratio = round(audio_input.size(-1) / audio_output.size(-1))
            nframes.floor_divide_(pooling_ratio)

            frame_counts.append(nframes.cpu())

        image_output = torch.cat(I_embeddings)
        audio_output = torch.cat(A_embeddings)
        nframes = torch.cat(frame_counts)
        S = compute_pooldot_similarity_matrix(
                image_output, audio_output, nframes)
        recalls = calc_recalls(S)
        A_r10 = recalls['A_r10']
        I_r10 = recalls['I_r10']
        M_r10 = (A_r10 + I_r10) / 2.
        A_r5 = recalls['A_r5']
        I_r5 = recalls['I_r5']
        M_r5 = (A_r5 + I_r5) / 2.
        A_r1 = recalls['A_r1']
        I_r1 = recalls['I_r1']
        M_r1 = (A_r1 + I_r1) / 2.
    return recalls
'''
    print(' * Audio R@10 {A_r10:.3f} / Image R@10 {I_r10:.3f}'
          ' / Mean R@10 {M_r10:.3f} over {N:d} validation pairs'.format(
          A_r10=A_r10, I_r10=I_r10, M_r10=M_r10, N=N_examples), flush=True)
    print(' * Audio R@5 {A_r5:.3f} / Image R@5 {I_r5:.3f}'
          ' / Mean R@5 {M_r5:.3f} over {N:d} validation pairs'.format(
          A_r5=A_r5, I_r5=I_r5, M_r5=M_r5, N=N_examples), flush=True)
    print(' * Audio R@1 {A_r1:.3f} / Image R@1 {I_r1:.3f}'
          ' / Mean R@1 {M_r1:.3f} over {N:d} validation pairs'.format(
          A_r1=A_r1, I_r1=I_r1, M_r1=M_r1, N=N_examples), flush=True)
'''
