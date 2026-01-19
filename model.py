import torch
import torch.nn as nn
import snntorch as snn
import torch.nn.functional as F



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# SNN-ResNet18
class SNNBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, beta=0.9):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        out = self.bn1(self.conv1(x))
        out, mem1 = self.lif1(out, mem1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out, mem2 = self.lif2(out, mem2)
        return out

class SNNResNet18(nn.Module):
    def __init__(self, num_classes=10, beta=0.9, num_steps=10):
        super().__init__()
        self.num_steps = num_steps
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.lif1 = snn.Leaky(beta=beta)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(64, 64, 2, 1, beta)
        self.layer2 = self._make_layer(64, 128, 2, 2, beta)
        self.layer3 = self._make_layer(128, 256, 2, 2, beta)
        self.layer4 = self._make_layer(256, 512, 2, 2, beta)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.lif_out = snn.Leaky(beta=beta, output=True)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride, beta):
        layers = []
        layers.append(SNNBasicBlock(in_channels, out_channels, stride, beta))
        for _ in range(1, blocks):
            layers.append(SNNBasicBlock(out_channels, out_channels, beta=beta))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()
        
        spk_rec = []
        mem_rec = []
        
        for step in range(self.num_steps):
            cur_x = x
            cur_x = self.bn1(self.conv1(cur_x))
            cur_x, mem1 = self.lif1(cur_x, mem1)
            cur_x = self.maxpool(cur_x)
            
            cur_x = self.layer1(cur_x)
            cur_x = self.layer2(cur_x)
            cur_x = self.layer3(cur_x)
            cur_x = self.layer4(cur_x)
            
            cur_x = self.avgpool(cur_x)
            cur_x = torch.flatten(cur_x, 1)
            cur_x = self.fc(cur_x)
            
            spk, mem_out = self.lif_out(cur_x, mem_out)
            spk_rec.append(spk)
            mem_rec.append(mem_out)
        
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)




