import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels_l, in_channels_g, attn_features):
        super().__init__()
        self.W_l = nn.Conv2d(in_channels_l, attn_features, kernel_size=1, bias=False)
        self.W_g = nn.Conv2d(in_channels_g, attn_features, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(attn_features, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.attn_features = attn_features
    
    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        x_l = self.W_l(x_l)
        x_g = self.W_g(x_g)
        x_g = F.interpolate(x_g, size=(H, W), mode='bilinear', align_corners=True)
        c = self.phi(self.relu(x_l + x_g))
        a = F.softmax(c.reshape(B, -1), dim=1).view(B, 1, H, W)
        f = x_l * a.expand_as(x_l)
        out = f.reshape(B, self.attn_features, -1).sum(dim=2)

        return a, out


class AttentionResNet34(nn.Module):
    def __init__(self, num_classes=2, attn_dropout=0.1, attn_features=256, get_attn=False):
        super().__init__()
        model = models.resnet34(num_classes=num_classes)
        self.stem = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool
        )
        self.block1 = nn.Sequential(
            model.layer1, model.layer2
        )
        self.block2 = model.layer3
        self.block3 = model.layer4
        self.attn1 = AttentionBlock(128, 512, attn_features)
        self.attn2 = AttentionBlock(256, 512, attn_features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # in_features for fc layer is concat of attn_features for attn1, attn2, layer4.
        self.fc = nn.Linear(2*attn_features + 512, num_classes)

        if attn_dropout > 0:
            self.attn_dropout = nn.Dropout(attn_dropout)
        else:
            self.attn_dropout = nn.Identity()
        self.get_attn = get_attn

    def forward(self, x):
        x = self.stem(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        g = self.avgpool(x3).view(x3.size(0), -1)
        a1, g1 = self.attn1(x1, x3)
        a2, g2 = self.attn2(x2, x3)
        g_hat = torch.cat([g, g1, g2], dim=1)
        g_hat = self.attn_dropout(g_hat)
        out = self.fc(g_hat)

        if self.get_attn:
            return out, (a1, a2)
        else:
            return out
        
    
class AttentionMobileNetV3(nn.Module):
    def __init__(self, num_classes=2, attn_dropout=0.1, attn_features=256, get_attn=False):
        super().__init__()

        model = models.mobilenet_v3_large(num_classes=num_classes)
        self.block1 = model.features[:5]
        self.block2 = model.features[5:10]
        self.block3 = model.features[10:]
        self.attn1 = AttentionBlock(40, 960, attn_features)
        self.attn2 = AttentionBlock(80, 960, attn_features)
        # in_features is concat of attn_features for attn1, attn2, block3.
        self.fc = nn.Sequential(
            nn.Linear(2 * attn_features + 960, 960), nn.ReLU(),
            nn.Linear(960, 480), nn.ReLU(),
            nn.Linear(480, num_classes)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if attn_dropout > 0:
            self.attn_dropout = nn.Dropout(attn_dropout)
        else:
            self.attn_dropout = nn.Identity()
        self.get_attn = get_attn

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        g = self.avgpool(x3).view(x3.size(0), -1)
        a1, g1 = self.attn1(x1, x3)
        a2, g2 = self.attn2(x2, x3)
        g_hat = torch.cat([g, g1, g2], dim=1)
        g_hat = self.attn_dropout(g_hat)
        out = self.fc(g_hat)

        if self.get_attn:
            return out, (a1, a2)
        else:
            return out
        

class AttentionInceptionV3(nn.Module):
    def __init__(self, num_classes=2, attn_dropout=0.3, attn_features=1024, get_attn=False):
        super().__init__()
        model = models.inception_v3(aux_logits=False, init_weights=False)
        model = nn.Sequential(*list(model.children())[:-3])
        self.block1 = model[:10]
        self.block2 = model[10:15]
        self.block3 = model[15:]
        self.attn1 = AttentionBlock(288, 2048, attn_features)
        self.attn2 = AttentionBlock(768, 2048, attn_features)
        self.fc = nn.Sequential(
            nn.Linear(2 * attn_features + 2048, 2048), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if attn_dropout > 0:
            self.attn_dropout = nn.Dropout(attn_dropout)
        else:
            self.attn_dropout = nn.Identity()
        self.get_attn = get_attn

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)

        g = self.avgpool(x3).view(x3.size(0), -1)
        a1, g1 = self.attn1(x1, x3)
        a2, g2 = self.attn2(x2, x3)
        g_hat = torch.cat([g, g1, g2], dim=1)
        g_hat = self.attn_dropout(g_hat)
        out = self.fc(g_hat)

        if self.get_attn:
            return out, (a1, a2)
        else:
            return out