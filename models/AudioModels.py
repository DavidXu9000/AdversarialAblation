import math
import numpy as np
import torch
import torch.nn as nn

def conv1d(in_planes, out_planes, width = 9, stride = 1, bias = False):
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), 
                     stride=stride, padding=(0,pad_amt), bias=bias)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, width=width, stride = stride)
        self.batchnorm1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width = width)
        self.batchnorm2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        out = self.relu(out)
        return out

class selfAttention(nn.Module):
    def __init__(self, inplanes, transform_planes = 64, dropout = .25):
        super(selfAttention, self).__init__()
        self.counter = 0
        self.dropout = dropout
        self.feature1 = nn.Conv2d(inplanes, transform_planes, kernel_size = (1, 1), bias = False, stride = 1)
        self.feature2 = nn.Conv2d(inplanes, transform_planes, kernel_size = (1, 1), bias = False, stride = 1)
        gamma = torch.Tensor(1,1)
        self.gamma = nn.Parameter(gamma)
        nn.init.ones_(gamma)

        self.softmax = nn.Softmax(dim = 2)

    #ablate random number of continuous channels such that the ablated area appears like a rectangle in the map
    def randomAblate(self, map):
        channels = map.size(1)
        start1 = np.random.randint(0, high=channels-1)
        end1 = np.random.randint(start1, high=channels-1)
        start2 = np.random.randint(0, high=channels-1)
        end2 = np.random.randint(start2, high=channels-1)

        tempMask = np.ones((channels, channels), dtype = np.float32)
        tempMask[start1:end1, start2:end2] = 0
        return tempMask, start1, end1, start2, end2

    def maxAbalate(self, map):
        values, indices = torch.max(map, dim = 2)
        channels = map.size(1)
        tempMask = np.ones((channels, channels), dtype = np.float32)
        tempMask[indices] = 0
        return tempMask

    def word_maxAblate(self, map, ranges):
        batch_size = map.size(0)
        channels = map.size(1)
        tmpMask = None
        flag = False
        for i in range(batch_size):
            tmp = None
            if ranges[i] != None:
                s = map[i,:,:]
                maxes, indices = torch.max(s, dim = 1)
                maxes.detach()
                indices.detach()

                max_word, row_index = torch.max(maxes.unsqueeze(0), dim=1)
                max_word.detach()
                row_index.detach()
                target_index = indices[row_index]

                try:
                    target_range = ranges[i][target_index * 16]
                    start = int(target_range[0] / 16)
                    end = int(target_range[1] / 16)
                    tmp = np.ones((1, channels, channels), dtype = np.float32)
                    tmp[start:end, :] = 0

                except KeyError:
                    self.counter += 1
                    tmp = np.ones((1, channels, channels), dtype = np.float32)
            else:
                tmp = np.ones((1, channels, channels), dtype = np.float32)
            if flag == False:
                tmpMask = tmp
                flag = True
            else:
                tmpMask = np.concatenate((tmpMask, tmp), axis = 0)
        return tmpMask

    def weightedDropout(self, map):
        batch_size = map.size(0)
        channels = map.size(1)
        num_params = channels * channels
        for i in range(batch_size):
            sliced_map = map[i, :, :]
            sliced_map = torch.reshape(sliced_map, (num_params, 1))
            sum = torch.sum(sliced_map)
            weights = torch.div(sliced_map, sum).numpy()
            samples = np.random.choice(num_params, num_params * self.dropout, False, weights)
            sliced_map[samples] = 0
            map[i,:,:] = torch.reshape(torch.from_numpy(), (channels, channels))


    def forward(self, x, ranges):
        #input is 1024 channel 1x128 image
        temp = x
        x = self.feature1(temp)
        y = self.feature2(temp)
        #feature space reduces to 64 channel 1x128 image

        temp = temp.squeeze(2)
        x = x.squeeze(2)
        y = y.squeeze(2)
        x = torch.transpose(x, 1, 2)
        x = torch.matmul(x, y)
        x = self.softmax(x) #self-attention map
         
        # t, start1, end1, start2, end2 = self.randomAblate(x)
        #t = self.word_maxAblate(x, ranges)
        #mask = torch.from_numpy(t).detach().to(x.device)
        #x = torch.mul(x,mask)

        x = torch.transpose(x, 1, 2)
        x = torch.matmul(temp, x)

        x = torch.mul(x, self.gamma)
        x = temp + x
        return x.unsqueeze(2)


class ResDavenetVQ(nn.Module):
    def probe(self, input_dim, num_words):
        layers = []
        layers.append(nn.Linear(input_dim, num_words))
        #layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def bigBlock(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )       
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, 
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1))
        return nn.Sequential(*layers)
            
    def __init__(self, feat_dim=40, block=BasicBlock, 
                 layers=[2, 2, 2, 2], layer_widths=[128, 128, 256, 512, 1024],
                 convsize=9, codebook_Ks=[512, 512, 512, 512, 512], 
                 commitment_cost=1, jitter_p=0.0, vqs_enabled=[0, 0, 0, 0, 0], 
                 EMA_decay=0.99, init_ema_mass=1, init_std=1, 
                 nonneg_init=False, num_vocab_words = -1, latent_dim = 64):
        super(ResDavenetVQ, self).__init__()
        self.layer_widths = layer_widths
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size = (self.feat_dim, 1), stride = 1, padding = (0,0), bias = False)
        self.batchnorm1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace = True)   

        self.block1 = self.bigBlock(block, layer_widths[1], layers[0], width = 9, stride = 2)
        self.block2 = self.bigBlock(block, layer_widths[2], layers[1], width = 9, stride = 2)
        self.block3 = self.bigBlock(block, layer_widths[3], layers[2], width = 9, stride = 2)
        self.block4 = self.bigBlock(block, layer_widths[4], layers[3], width = 9, stride = 2)
        self.num_vocab_words = num_vocab_words


        if num_vocab_words > 0:
            for layer in [self.batchnorm1, self.relu, self.conv1, self.block1, self.block2, self.block3, self.block4]:
                for p in layer.parameters():
                        p.requires_grad = False

            self.probes = nn.ModuleList([self.probe(layer_widths[j+1], num_vocab_words) for j in range(4)])

        #self.attention = selfAttention(layer_widths[4], transform_planes = 64)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, nframes, ranges, probe = False, latent_dim = 64):
        #Instead we can simply define the probes here as well so we don't need to compute the embeddings outside of the model
        #Only need to compute the answer which is much easier to do
        #This will return a boolean vector of predictions

        n = x.size(0)
        predictions = torch.zeros((n, 4, self.num_vocab_words))
        predictions = predictions.to(x.device)
        pooling = [nn.AdaptiveAvgPool2d((self.layer_widths[j+1], 1)) for j in range(4)]

        if(x.dim() == 3):
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.block1(x)
        if probe == True:
            pooled = []
            for j in range(n):
                nF = max(1, nframes[j] // 2)
                pooled.append(pooling[0](x[j][:,:,0:nF].squeeze().unsqueeze(0)).squeeze(0).transpose(0,1))
            pooled = torch.cat(pooled, axis = 0).squeeze()
            predictions[:, 0, :] = self.probes[0](pooled)

        x = self.block2(x)
        if probe == True:
            pooled = []
            for j in range(n):
                nF = max(1, nframes[j] // 4)
                pooled.append(pooling[1](x[j][:,:,0:nF].squeeze().unsqueeze(0)).squeeze(0).transpose(0,1))

            pooled = torch.cat(pooled, axis = 0).squeeze()
            predictions[:, 1, :] = self.probes[1](pooled)

        x = self.block3(x)
        if probe == True:
            pooled = []
            for j in range(n):
                nF = max(1, nframes[j] // 8)
                pooled.append(pooling[2](x[j][:,:,0:nF].squeeze().unsqueeze(0)).squeeze(0).transpose(0,1))

            pooled = torch.cat(pooled, axis = 0).squeeze()
            predictions[:, 2, :] = self.probes[2](pooled)

        x = self.block4(x)
        if probe == True:
            pooled = []
            for j in range(n):
                nF = max(1, nframes[j] // 16)
                pooled.append(pooling[3](x[j][:,:,0:nF].squeeze().unsqueeze(0)).squeeze(0).transpose(0,1))
                #pooled.append(torch.mean(x[j][:,:,0:nF].squeeze(), axis = 1).unsqueeze(0))

            pooled = torch.cat(pooled, axis = 0).squeeze()
            predictions[:, 3, :] = self.probes[3](pooled)

            return predictions



        #x = self.attention(x, ranges)
        #x = x.squeeze(2)
        return x

