# Author: David Harwath

import math
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_recalls(S):
    """
    Computes recall at 1, 5, and 10 given a similarity matrix S.
    By convention, rows of S are assumed to correspond to images and columns are captions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    if isinstance(S, torch.autograd.Variable):
        S = S.data
    n = S.size(0)
    A2I_scores, A2I_ind = S.topk(10, 0)
    I2A_scores, I2A_ind = S.topk(10, 1)
    A_r1 = AverageMeter()
    A_r5 = AverageMeter()
    A_r10 = AverageMeter()
    I_r1 = AverageMeter()
    I_r5 = AverageMeter()
    I_r10 = AverageMeter()
    for i in range(n):
        A_foundind = -1
        I_foundind = -1
        for ind in range(10):
            if A2I_ind[ind, i] == i:
                I_foundind = ind
            if I2A_ind[i, ind] == i:
                A_foundind = ind
        # do r1s
        if A_foundind == 0:
            A_r1.update(1)
        else:
            A_r1.update(0)
        if I_foundind == 0:
            I_r1.update(1)
        else:
            I_r1.update(0)
        # do r5s
        if A_foundind >= 0 and A_foundind < 5:
            A_r5.update(1)
        else:
            A_r5.update(0)
        if I_foundind >= 0 and I_foundind < 5:
            I_r5.update(1)
        else:
            I_r5.update(0)
        # do r10s
        if A_foundind >= 0 and A_foundind < 10:
            A_r10.update(1)
        else:
            A_r10.update(0)
        if I_foundind >= 0 and I_foundind < 10:
            I_r10.update(1)
        else:
            I_r10.update(0)

    recalls = {'A_r1':A_r1.avg, 'A_r5':A_r5.avg, 'A_r10':A_r10.avg,
                'I_r1':I_r1.avg, 'I_r5':I_r5.avg, 'I_r10':I_r10.avg}
                #'A_meanR':A_meanR.avg, 'I_meanR':I_meanR.avg}

    return recalls

def computeMatchmap(I, A):
    assert(I.dim() == 3)
    assert(A.dim() == 2)
    D = I.size(0)
    H = I.size(1)
    W = I.size(2)
    T = A.size(1)                                                                                                                     
    Ir = I.view(D, -1).t()
    matchmap = torch.mm(Ir, A)
    matchmap = matchmap.view(H, W, T)  
    return matchmap

def matchmapSim(M, simtype):
    assert(M.dim() == 3)
    if simtype == 'SISA':
        return M.mean()
    elif simtype == 'MISA':
        M_maxH, _ = M.max(0)
        M_maxHW, _ = M_maxH.max(0)
        return M_maxHW.mean()
    elif simtype == 'SIMA':
        M_maxT, _ = M.max(2)
        return M_maxT.mean()
    else:
        raise ValueError

def sampled_margin_rank_loss(image_outputs, audio_outputs, nframes, margin=1., simtype='MISA'):
    """
    Computes the triplet margin ranking loss for each anchor image/caption pair
    The impostor image/caption is randomly sampled from the minibatch
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    loss = torch.zeros(1, device=image_outputs.device, requires_grad=True)
    for i in range(n):
        I_imp_ind = i
        A_imp_ind = i
        while I_imp_ind == i:
            I_imp_ind = np.random.randint(0, n)
        while A_imp_ind == i:
            A_imp_ind = np.random.randint(0, n)
        nF = nframes[i]
        nFimp = nframes[A_imp_ind]
        anchorsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[i][:, 0:nF]), simtype)
        Iimpsim = matchmapSim(computeMatchmap(image_outputs[I_imp_ind], audio_outputs[i][:, 0:nF]), simtype)
        Aimpsim = matchmapSim(computeMatchmap(image_outputs[i], audio_outputs[A_imp_ind][:, 0:nFimp]), simtype)
        A2I_simdif = margin + Iimpsim - anchorsim
        if (A2I_simdif.data > 0).all():
            loss = loss + A2I_simdif
        I2A_simdif = margin + Aimpsim - anchorsim
        if (I2A_simdif.data > 0).all():
            loss = loss + I2A_simdif
    loss = loss / n
    return loss

def compute_matchmap_similarity_matrix(image_outputs, audio_outputs, nframes, simtype='MISA'):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 3)
    n = image_outputs.size(0)
    S = torch.zeros(n, n, device=image_outputs.device)
    for image_idx in range(n):
            for audio_idx in range(n):
                nF = max(1, nframes[audio_idx])
                S[image_idx, audio_idx] = matchmapSim(computeMatchmap(image_outputs[image_idx], audio_outputs[audio_idx][:, 0:nF]), simtype)
    return S

def compute_maxes(image_outputs, audio_outputs, num_words_drop, nframes, ablate, ranges, range_dictionary, id, find_ss = True):
    n = image_outputs.size(0)
    imagePool = nn.AdaptiveAvgPool2d((1,1))
    pooled_image_outputs = imagePool(image_outputs).squeeze(3).squeeze(2) #32*1024
    audioPool = nn.AdaptiveAvgPool2d((1,1))
    
    max_times = []
    word_ss = {}
    for i in range(n):
        pooled_audio_list = []
        timestamps = []
        word_list = []
        try:
            for k in range(15, 2048, 16):
                start_frame = k - 7
                end_frame = start_frame + 1
                start_pos = start_frame // 16
                end_pos = start_pos + 1

                assert(start_pos < 128 and start_pos >= 0)
                assert(end_pos <= 128)
                pooled_audio_list.append(audioPool(audio_outputs[i][:, :, start_pos:end_pos]).unsqueeze(0))
                timestamps.append((start_pos * 16, start_pos * 16 + 1))
                
            pooled_audio_outputs = torch.cat(pooled_audio_list).squeeze(3).squeeze(2)
            S = torch.mm(pooled_image_outputs[i].unsqueeze(0), pooled_audio_outputs.t())

            try:  
                if find_ss:
                    for timestamp in range_dictionary[id[i]]:
                        start_frame = ranges[i][timestamp][0]
                        end_frame = ranges[i][timestamp][1]
                        word = ranges[i][timestamp][2]
                        start_pos = round(start_frame/16)
                        end_pos = round(end_frame/16)
                        if start_pos == end_pos:
                            end_pos += 1
                        if word not in word_ss:
                            val, ind = torch.max(S[:, start_pos:end_pos], axis = 1)
                            word_ss[word] = [val.item(), 1]
                        else:
                            val, ind = torch.max(S[:, start_pos:end_pos], axis = 1)
                            word_ss[word][0] += val.item()
                            word_ss[word][1] += 1
            except KeyError:
                pass
                     
            sorted_order, indices = torch.sort(S, dim=1, descending=True)
            
            indices = indices.squeeze(0)
            num_words_added = 0
            for j in range(len(timestamps)):
                if num_words_added == num_words_drop:
                    break
                found_word = True
                for k in range(len(word_list)):
                    top_start = word_list[k][0]
                    curr_start = timestamps[indices[j]][0]
                    if abs(top_start - curr_start) < ablate:
                        found_word = False

                if not found_word:
                    continue

                word_list.append(timestamps[indices[j]])
                num_words_added += 1
            max_times.append(word_list)
        except KeyError:
            for j in range(min(num_words_drop, len(timestamps))):
                word_list.append((-1,-1))
            max_times.append(word_list)
    return max_times, word_ss



def compute_max_times(image_outputs, audio_outputs, num_words_drop, nframes, ablate):
    n = image_outputs.size(0)
    imagePool = nn.AdaptiveAvgPool2d((1,1))
    pooled_image_outputs = imagePool(image_outputs).squeeze(3).squeeze(2) #32*1024
    audioPool = nn.AdaptiveAvgPool2d((1,1))
    
    max_times = []
    for i in range(n):
        pooled_audio_list = []
        timestamps = []
        word_list = []
        try:
            for k in range(15, 2048, 16):
                start_frame = k - 7
                end_frame = start_frame + 1
                start_pos = start_frame // 16
                end_pos = start_pos + 1

                assert(start_pos < 128 and start_pos >= 0)
                assert(end_pos <= 128)
                pooled_audio_list.append(audioPool(audio_outputs[i][:, :, start_pos:end_pos]).unsqueeze(0))
                timestamps.append((start_pos * 16, start_pos * 16 + 1))
                
            pooled_audio_outputs = torch.cat(pooled_audio_list).squeeze(3).squeeze(2)
            S = torch.mm(pooled_image_outputs[i].unsqueeze(0), pooled_audio_outputs.t())
            sorted_order, indices = torch.sort(S, dim=1, descending=True)
            
            indices = indices.squeeze(0)
            num_words_added = 0
            for j in range(len(timestamps)):
                if num_words_added == num_words_drop:
                    break
                found_word = True
                for k in range(len(word_list)):
                    top_start = word_list[k][0]
                    curr_start = timestamps[indices[j]][0]
                    if abs(top_start - curr_start) < ablate:
                        found_word = False

                if not found_word:
                    continue

                word_list.append(timestamps[indices[j]])
                num_words_added += 1
            max_times.append(word_list)
        except KeyError:
            for j in range(min(num_words_drop, len(timestamps))):
                word_list.append((-1,-1))
            max_times.append(word_list)
    return max_times

def compute_pooldot_similarity_matrix(image_outputs, audio_outputs, nframes):
    """
    Assumes image_outputs is a (batchsize, embedding_dim, rows, height) tensor
    Assumes audio_outputs is a (batchsize, embedding_dim, 1, time) tensor
    Returns similarity matrix S where images are rows and audios are along the columns
    S[i][j] is computed as the dot product between the meanpooled embeddings of
    the ith image output and jth audio output
    """
    assert(image_outputs.dim() == 4)
    assert(audio_outputs.dim() == 4)
    n = image_outputs.size(0)
    imagePoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_image_outputs = imagePoolfunc(image_outputs).squeeze(3).squeeze(2)
    audioPoolfunc = nn.AdaptiveAvgPool2d((1, 1))
    pooled_audio_outputs_list = []
    for idx in range(n):
        nF = max(1, nframes[idx])
        pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
    pooled_audio_outputs = torch.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
    S = torch.mm(pooled_image_outputs, pooled_audio_outputs.t())
    return S

def one_imposter_index(i, N):
    imp_ind = random.randint(0, N - 2)
    if imp_ind == i:
        imp_ind = N - 1
    return imp_ind

def basic_get_imposter_indices(N):
    imposter_idc = []
    for i in range(N):
        # Select an imposter index for example i:
        imp_ind = one_imposter_index(i, N)
        imposter_idc.append(imp_ind)
    return imposter_idc

def semihardneg_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S
    Output: The one-way triplet loss from rows of S to columns of S. Impostors are taken
    to be the most similar point to the anchor that is still less similar to the anchor
    than the positive example.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    positive_scores = S.diag()
    mask = ((S - S.diag().view(-1,1)) < 0).float().detach()
    imposter_scores = (S * mask).max(dim=1).values
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()
    return loss

def sampled_triplet_loss_from_S(S, margin):
    """
    Input: Similarity matrix S
    Output: The one-way triplet loss from rows of S to columns of S. Imposters are
    randomly sampled from the columns of S.
    You would need to run this function twice, once with S and once with S.t(),
    in order to compute the triplet loss in both directions.
    """
    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    N = S.size(0)
    positive_scores = S.diag()
    imp_indices = np.random.randint(0, N-1, size=N)
    for j, ind in enumerate(imp_indices):
        if ind >= j:
            imp_indices[j] = ind + 1
    imposter_scores = S[range(N), imp_indices]
    loss = (imposter_scores - positive_scores + margin).clamp(min=0).mean()
    return loss

def masked_margin_softmax(S, margin, mask):

    assert(S.dim() == 2)
    assert(S.size(0) == S.size(1))
    
    target = torch.LongTensor(list(range(S.size(0)))).to(S.device)
    deltas = margin * torch.eye(S.size(0)).to(S.device)
    S = S - deltas

    I2C_loss = F.nll_loss(masked_softmax(S, mask), target)

    S = S.t()
    C2I_loss = F.nll_loss(masked_softmax(S, mask), target)

    return I2C_loss + C2I_loss

def masked_softmax(S, mask):
    maxes = torch.max(S, 1, keepdim=True)[0]
    masked_s_exp = torch.exp(S - maxes) * mask 
    norm = torch.sum(masked_s_exp, 1, keepdim = True) + 1e-11
    log_prob = S - maxes - torch.log(norm)
    return log_prob

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(base_lr, lr_decay, lr_decay_multiplier, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed every lr_decay epochs"""
    lr = base_lr * (lr_decay_multiplier ** (epoch // lr_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def load_progress(prog_pkl, quiet=False):
    """
    load progress pkl file
    Args:
        prog_pkl(str): path to progress pkl file
    Return:
        progress(list):
        epoch(int):
        global_step(int):
        best_epoch(int):
        best_avg_r10(float):
    """
    def _print(msg):
        if not quiet:
            print(msg)

    with open(prog_pkl, "rb") as f:
        prog = pickle.load(f)
        epoch, global_step, best_epoch, best_avg_r10, _ = prog[-1]

    _print("\nPrevious Progress:")
    msg =  "[%5s %7s %5s %7s %6s]" % ("epoch", "step", "best_epoch", "best_avg_r10", "time")
    _print(msg)
    return prog, epoch, global_step, best_epoch, best_avg_r10
