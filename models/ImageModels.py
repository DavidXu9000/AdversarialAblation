import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class Resnet50(models.ResNet):
    def __init__(self, embedding_dim = 1024, pretrained = True):
        super(Resnet50, self).__init__(models.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained == True:
            url = models.resnet.model_urls['resnet50']
            self.load_state_dict(model_zoo.load_url(url))
        self.avgpool = None
        self.fc = None
        self.embed = nn.Conv2d(2048, embedding_dim, kernel_size = 1, padding = 1, stride = 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embed(x)
        return x