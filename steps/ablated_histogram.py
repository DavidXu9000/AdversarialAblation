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
from collections import defaultdict


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


def compute_stats(audio_model, image_model, train_loader, val_loaders, args, exp_dir, resume, timestamps, splice, num_words_drop, word_dropout, unsupervised, ablate):
    batch_timer = AverageMeter()
    global_step, epoch = 0,0
    f = open(timestamps, "r")
    prev = ""
    currTree = None
    ranges = {}
    range_dictionary = {}
    timestamp_set = set()
    for timestamp in f:
        components = timestamp.split(" ")
        uttid = components[0]
        if uttid != prev:
            currTree = RangeTree()
            timestamp_set = set()
        prev = uttid
        start_time = float(components[2])
        end_time = start_time + float(components[3])
        word = components[4].lower()
        start_frame = round(start_time * 100) 
        end_frame = round(end_time * 100)

        currTree[start_frame:end_frame] = (start_frame, end_frame, word)
        timestamp_set.add(start_time)
        ranges[uttid] = currTree
        range_dictionary[uttid] = timestamp_set
    
    # Set device and maybe load snapshot
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(True)

    print('Found %d GPUs' % torch.cuda.device_count())
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    if not isinstance(image_model, torch.nn.DataParallel):
        image_model = nn.DataParallel(image_model)
    
    audio_model = audio_model.to(device)
    image_model = image_model.to(device)

    ablated_count = defaultdict(int)
    avg_max_ss = {}

    torch.cuda.empty_cache()
    batch_time = time.time()
    for i, (image_input, audio_input, nframes, strings, id) in enumerate(train_loader): 
        B = audio_input.size(0)
        T = audio_input.size(-1)
        audio_input = audio_input.to(device)
        image_input = image_input.to(device)

        batch_ranges = []
        for j in range(len(id)):
            try:
                batch_ranges.append(ranges[id[j]])
            except KeyError:
                batch_ranges.append(None)

        with torch.no_grad():
            image_output = image_model(image_input)
            audio_output = audio_model(audio_input, nframes, None)
        
            start_time = time.time()
            max_times, word_ss = compute_maxes(image_output, audio_output, num_words_drop, nframes, ablate, batch_ranges, range_dictionary, id)
            for j in range(len(max_times)):
                timestamp_list = max_times[j]
                timestamp_list.sort(reverse=True)
                for k in range(min(num_words_drop, len(timestamp_list))):
                    timestamp = timestamp_list[k]
                    if id[j] in ranges:
                        word_start_time = timestamp[0]
                        if word_start_time in ranges[id[j]]:
                            word_ablated = ranges[id[j]][word_start_time][2]
                            ablated_count[word_ablated] += 1 

            for word in word_ss:
                if word not in avg_max_ss:
                    avg_max_ss[word] = word_ss[word]
                else:
                    avg_max_ss[word][0] += word_ss[word][0]
                    avg_max_ss[word][1] += word_ss[word][1]


        del (image_output, audio_output)
        if global_step % args.n_print_steps == 0:
            print('Epoch: [{0}][{1}/{2}]'
                  ' Time = {bt.val:.3f} ({bt.avg:.3f})'.format(
                  epoch, i, len(train_loader), bt=batch_timer), flush=True)
        batch_timer.update(time.time() - batch_time)
        batch_time = time.time()
        global_step += 1

    for word in avg_max_ss:
        avg_max_ss[word] = avg_max_ss[word][0] / avg_max_ss[word][1]

    return ablated_count, avg_max_ss

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
