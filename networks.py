import torch
import torch.nn as nn
import copy

class CAE_3_bn(nn.Module):
    
    def __init__(self, input_channels, out_channels, kernel_size, leaky = True, out_padding = False):
        super(CAE_3_bn, self).__init__()
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        if leaky:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
            
        if out_padding:
            pad = 1
        else:
            pad = 0
        
        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = 2)
        self.bn1_1 = nn.BatchNorm1d(num_features = out_channels, eps = 1e-5,
                                    momentum = 0.1)
        self.conv2 = nn.Conv1d(in_channels = out_channels, 
                               out_channels = out_channels*2,
                               kernel_size = kernel_size, stride = 2)
        self.bn2_1 = nn.BatchNorm1d(num_features = out_channels*2, eps = 1e-5,
                                    momentum = 0.1)
        self.conv3 = nn.Conv1d(in_channels = out_channels*2, 
                               out_channels = out_channels*4,
                               kernel_size = kernel_size, stride = 1)
        self.bn3_1 = nn.BatchNorm1d(num_features = out_channels*4, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = out_channels*4, 
                                          out_channels = out_channels*2,
                                          kernel_size = kernel_size, stride = 1)
        self.bn3_2 = nn.BatchNorm1d(num_features = out_channels*2, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = out_channels*2, 
                                          out_channels = out_channels,
                                          kernel_size = kernel_size, stride = 2, output_padding = 1)
        self.bn2_2 = nn.BatchNorm1d(num_features = out_channels, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv1 = nn.ConvTranspose1d(in_channels = out_channels, 
                                          out_channels = input_channels, kernel_size = kernel_size, 
                                          stride = 2, output_padding = 1)
        self.bn1_2 = nn.BatchNorm1d(num_features = input_channels, eps = 1e-5,
                                    momentum = 0.1)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        
    def encoder(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        
        return x
    
    def decoder(self, x):
        x = self.deconv3(x)
        x = self.relu1_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu2_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        x = self.relu3_2(x)
        x = self.bn1_2(x)
        
        return x
    
    def forward(self, x):
        bottleneck = self.encoder(x)
        reconst = self.decoder(bottleneck)
        
        return reconst, bottleneck
    
    
class Classifier(nn.Module):
    
    def __init__(self, network, leaky = True):
        super(Classifier, self).__init__()
        self.encoder = network.encoder
        self.fc1 = nn.Linear(in_features = 1024, out_features = 300)
        self.fc2 = nn.Linear(in_features = 300, out_features = 22)
        
        if leaky:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.relu1_1(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu2_1(x)
        x = self.fc2(x)
        
        
        return x
