import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import sys

class Scannet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, pretrained=False,train_model=True):
        super(Scannet, self).__init__()
        self.train_model=train_model
        self.pretrained=pretrained
        vgg =models.vgg16(pretrained)
        features = list(vgg.features.children())
        # Set padding=0 in features
        for layer in features:
            layer.padding = (0, 0)
        self.features = nn.Sequential(*features)
        # repalce the FC layer of vgg 166
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=2, stride=1, padding=0),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=1024, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        '''
        initialize_weights for new conv2 as kaiming suggested
        '''
        def __kaiming_init_(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        if not self.pretrained:
            for m in self.features:
                __kaiming_init_(m)
        for m in self.classifier:
            __kaiming_init_(m)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.train_model:
            x = F.softmax(x)[:,1].cpu()
        return x
