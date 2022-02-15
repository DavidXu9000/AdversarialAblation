# Author: David Harwath, Wei-Ning Hsu
import argparse
import numpy as np
import os
import pickle
import shutil
import sys
import time
import torch
from collections import OrderedDict

import dataloaders
from steps.traintest import train, validate
from steps.predict_word_presence import predict_word_presence
from run_utils import str2bool, set_seeds, create_audio_model, create_image_model, load_state_dict

def get_count(dictionary, word):
    if word in dictionary:
        return dictionary[word]
    else:
        return 0
def load_args(old_args, exp_dir):
    """
    If resuming, load args.pkl from the experiment directory, and overwrite
    `data_train`/`data_val`/`resume`.
    """
    print('loading arguments from %s/args.pkl' % exp_dir)
    with open('%s/args.pkl' % exp_dir, 'rb') as f:
        tmp_args = pickle.load(f)
    for k in vars(old_args):
        if hasattr(tmp_args, k):
            setattr(old_args, k, getattr(tmp_args, k))
        else:
            print('...missing arg: %s' % k)
    return old_args


def load_dataloaders(data_train, data_val, timestamps, batch_size, num_workers):
    print('loading data from %s / %s' % (data_train, data_val))
    train_dset = dataloaders.ImageCaptionDataset(data_train, timestamps)
    train_loader = torch.utils.data.dataloader.DataLoader(
            train_dset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True)

    val_loaders = []

    for i in range(5):
        for j in range(5):
            val_dset = dataloaders.ImageCaptionDataset(
                data_val, timestamps, image_conf={'center_crop':True}, val_num = i, cap_num = j)
            val_loader = torch.utils.data.dataloader.DataLoader(
            val_dset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True)

            val_loaders.append(val_loader)

    return train_loader, val_loaders

def load_state_dicts(audio_model, image_model, seed_dir, seed_epoch, name = None, baseline = False):
    if name != None:
        print("loading parameters")
        if baseline == True:
            audio_states = torch.load("/data2/scratch/dx789/best/SpokenCOCO/baseline/%s-audio.pth" % name)
            image_states = torch.load("/data2/scratch/dx789/best/SpokenCOCO/baseline/%s-image.pth" % name)
        else:
            audio_states = torch.load("/home/dx789/secondRun/VisualAudio/scripts/exps/%s/models/best_audio_model.pth" % name)
            image_states = torch.load("/home/dx789/secondRun/VisualAudio/scripts/exps/%s/models/best_image_model.pth" % name)

            '''
            audio_states = torch.load("/data1/scratch/dx789/best/SpokenCOCO/supervised/%s-audio.pth" % name)
            image_states = torch.load("/data1/scratch/dx789/best/SpokenCOCO/supervised/%s-image.pth" % name)
            '''


    else:
        if seed_epoch > 0:
            audio_states = torch.load(
                    '%s/models/audio_model.e%d.pth' % (seed_dir, seed_epoch))
            image_states = torch.load(
                    '%s/models/image_model.e%d.pth' % (seed_dir, seed_epoch))
        else:
            audio_states = torch.load(
                    '%s/models/best_audio_model.pth' % seed_dir)
            image_states = torch.load(
                    '%s/models/best_image_model.pth' % seed_dir)

    load_state_dict(audio_model, audio_states)
    load_state_dict(image_model, image_states)
    print('loaded parameters from %s/models/' % seed_dir)


def get_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ResDavenet args
    parser.add_argument('--audio-model', type=str, default='ResDavenetVQ', 
            choices=['ResDavenetVQ'], help='audio model architecture')
    parser.add_argument('--image-model', type=str, default='Resnet50', 
            choices=['Resnet50'], help='image model architecture')
    parser.add_argument('--freeze-image-model', type=str2bool, default=False,
            help='Freeze image model parameters.')
    parser.add_argument('--pretrained-image-model', type=str2bool, default=True, 
            help='Use an image network pretrained on ImageNet')
    parser.add_argument('--seed-dir', type=str, default='',
            help=('Load image and audio model weights from a seed model. Overrides' 
                  ' using an image model pretrained on ImageNet'))
    parser.add_argument('--seed-epoch', type=int, default=-1, 
            help='Load snapshot from this epoch')
    parser.add_argument('--margin', type=float, default=1.0, 
            help='Margin paramater for triplet loss')
    parser.add_argument('--layer-widths', type=str, default='128,256,256,512,1024', 
            help='ResDavenet layer/block sizes')
    parser.add_argument('--layer-depths', type=str, default='2,2,2,2', 
            help='ResDavenet depth of each residual block')
    parser.add_argument('--convsize', type=int, default=9,
            help='ResDavenet 1-D convolution width')
    parser.add_argument('--seed', type=int, default=8675309, help='Random seed')
    
    # VQ args
    parser.add_argument('--VQ-commitment-cost', default=1, type=float, 
            help='VQ commitment cost')
    parser.add_argument('--VQ-turnon', type=str, default='0,0,0,0,0', 
            help=('Comma-separated list of integers representing which VQ layers' 
                  ' are turned on.'))
    parser.add_argument('--VQ-sizes', type=str, default='1024,1024,1024,1024,1024', 
            help=('Comma-separated list of integers representing the codebook sizes' 
                  ' for the quantization layers.'))
    parser.add_argument('--nonneg-init', type=str2bool, default=False,
            help='Clamp the initial VQ codebooks at 0')
    parser.add_argument('--init-std', default=1, type=float, 
            help='VQ codebook initialization standard deviation')
    parser.add_argument('--init-ema-mass', default=1, type=float,
            help='EMA mass for the initial codebook')
    parser.add_argument('--jitter', type=float, default=0.12, 
            help='Temporal jitter probability (equal for both left and right)')

    return parser
    

def get_train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # training and optimization args
    parser.add_argument('--splice', type=int, default=0)
    parser.add_argument('--num-words-dropout', type=int, default=2)
    parser.add_argument('--word-dropout', type=float, default=0.1)
    parser.add_argument('--unsupervised', type=int, default=1)
    parser.add_argument('--ablate', type=int, default=50)
    parser.add_argument('--optim', type=str, default='adam',
            help='training optimizer', choices=['sgd', 'adam'])
    parser.add_argument('--num-probes', type=int, default = 1000)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
            help='mini-batch size')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR', 
            help='initial learning rate')
    parser.add_argument('--lr-decay', type=int, default=3, metavar='LRDECAY',
            help=('Multiply the learning rate by lr-decay-multiplier every lr-decay'
                  ' number of epochs'))
    parser.add_argument('--lr-decay-multiplier', type=float, default=0.95,
            metavar='LRDECAYMULT',
            help='Multiply the learning rate by this factor every lr-decay epochs')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-7, metavar='W', 
            help='weight decay')
    parser.add_argument('--force-start-epoch', type=int, default=0, 
            metavar='force_start_epoch', 
            help=('Start on this epoch number (for controlling the position in the'
                  ' learning rate schedule)'))
    parser.add_argument('--n-epochs', type=int, default=150,
            help='number of maximum training epochs')
    parser.add_argument('--n-print-steps', type=int, default=100,
            help='number of steps to print statistics')
    parser.add_argument('--save-every', type=int, default=10,
            help=('Keep a model checkpoint every this many epochs. Set to -1 to'
                  ' deactivate'))

    return parser


def get_run_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # I/O args
    parser.add_argument('--data-train', type=str, default='',
            help='training data json')
    parser.add_argument('--data-val', type=str, default='',
            help='validation data json')
    parser.add_argument('--exp-dir', type=str, default='',
            help='directory to dump experiments')
    parser.add_argument('--resume', type=str2bool, default=False,
            help='load from exp_dir if True')
    parser.add_argument('--mode', type=str, default='eval',
            choices=['train', 'eval', 'probe', 'stats'],
            help='Train the model; otherwise, perform validation')
    parser.add_argument('--num-workers', type=int, default=8,
            help='number of dataloading workers')

    return parser


if __name__ == '__main__':
    print('I am process %s, running on %s: starting (%s)' % (
            os.getpid(), os.uname()[1], time.asctime()))
    
    parser = get_default_parser()
    parser = get_train_parser(parser)
    parser = get_run_parser(parser)
    
    args = parser.parse_args()
    set_seeds(args.seed)

    def get_and_del_attr(name):
        val = getattr(args, name)
        delattr(args, name)
        return val

    exp_dir = get_and_del_attr('exp_dir')

    resume = get_and_del_attr('resume')
    data_train = get_and_del_attr('data_train')
    data_val = get_and_del_attr('data_val')
    data_mt = os.getenv('data_mt')

    mode = get_and_del_attr('mode')
    if resume:
        args = load_args(args, exp_dir)

    for k in vars(args):
        print('%-40s : %s' % (k, getattr(args, k)))
    
    train_loader, val_loaders = load_dataloaders(
            data_train, data_val, data_mt, args.batch_size, args.num_workers)
    
    if mode == 'probe':
        audio_model = create_audio_model(
                args.audio_model, args.VQ_sizes, args.layer_widths, args.layer_depths,
                args.VQ_turnon, args.convsize, args.VQ_commitment_cost,
                args.jitter, args.init_ema_mass, args.init_std, args.nonneg_init, args.num_probes)
    else:
        audio_model = create_audio_model(
                args.audio_model, args.VQ_sizes, args.layer_widths, args.layer_depths,
                args.VQ_turnon, args.convsize, args.VQ_commitment_cost,
                args.jitter, args.init_ema_mass, args.init_std, args.nonneg_init)


    image_model = create_image_model(args.image_model, args.pretrained_image_model)
    
    # Start Training
    if mode == 'train':
        if args.seed_dir:
            load_state_dicts(audio_model, image_model,
                             args.seed_dir, args.seed_epoch)
    
        if not resume:
            print('\nCreating experiment directory: %s' % exp_dir)
            os.makedirs('%s/models' % exp_dir)
            with open('%s/args.pkl' % exp_dir, 'wb') as f:
                pickle.dump(args, f)
        
        train(audio_model, image_model, train_loader, val_loaders,
              args, exp_dir, resume, data_mt, args.splice==1, args.num_words_dropout, args.word_dropout, args.unsupervised==1, args.ablate)
    elif mode == 'eval':
        name = "%d-dropout-%d-splice-%d-num-words-2-train" % (int(args.word_dropout * 100), args.splice, args.num_words_dropout) 
        load_state_dicts(audio_model, image_model, exp_dir, -1, name=name, baseline = True)

        avg_acc = 0
        avg_acc_5 = 0
        avg_acc_1 = 0
        for k in range(len(val_loaders)):
            recalls = validate(audio_model, image_model, val_loaders[k], args)
            acc = (recalls['A_r10'] + recalls['I_r10']) / 2
            acc_5 = (recalls['A_r5'] + recalls['I_r5']) / 2
            acc_1 = (recalls['A_r1'] + recalls['I_r1']) / 2
            avg_acc += acc
            print(acc)
            avg_acc_5 += acc_5
            avg_acc_1 += acc_1
        avg_acc /= len(val_loaders)
        avg_acc_5 /= len(val_loaders)
        avg_acc_1 /= len(val_loaders)

        print("R@10 = %f, R@5 = %f, R@1 = %f" % (avg_acc, avg_acc_5, avg_acc_1))
    
    elif mode == 'probe':
        name = "%d-dropout-%d-splice-%d-num-words-2-train" % (int(args.word_dropout * 100), args.splice, args.num_words_dropout) 
        load_state_dicts(audio_model, image_model, exp_dir, -1, name=name, baseline = (int(args.word_dropout * 100) == 0))
        
        if not resume:
            print('\nCreating experiment directory: %s' % exp_dir)
            os.makedirs('%s/models' % exp_dir)
            with open('%s/args.pkl' % exp_dir, 'wb') as f:
                pickle.dump(args, f)

        acc = predict_word_presence(audio_model, train_loader, val_loaders, "/home/harwath/data/SpokenCOCO/train_ctm.txt", "/home/harwath/data/SpokenCOCO/val_ctm.txt", args.n_epochs, args, exp_dir)
        print(acc)
    else:
        raise ValueError('Unsupported mode %s' % mode)
