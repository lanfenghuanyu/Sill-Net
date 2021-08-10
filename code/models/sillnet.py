from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import cv2
import numpy as np

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()
   
    def forward(self, x):
        numel = x.numel() / x.shape[0]
        return x.view(-1, int(numel)) 

def convNoutput(convs, input_size): # predict output size after conv layers
    input_size = int(input_size)
    input_channels = convs[0][1].weight.shape[1] # input channel
    output = torch.Tensor(1, input_channels, input_size, input_size)
    with torch.no_grad():
        for conv in convs:
            output = conv(output)
    return output.numel(), output.shape

class stn(nn.Module):
    def __init__(self, input_channels, input_size, params):
        super(stn, self).__init__()
        
        self.input_size = input_size

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
                    nn.ReplicationPad2d(2),
                    nn.Conv2d(input_channels, params[0], kernel_size=5, stride=1),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )
        self.conv2 = nn.Sequential(
                    nn.ReplicationPad2d(2),
                    nn.Conv2d(params[0], params[1], kernel_size=5, stride=1),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                    
        self.conv3 = nn.Sequential(
                    nn.ReplicationPad2d(2),
                    nn.Conv2d(params[1], params[2], kernel_size=3, stride=1),
                    nn.ReLU(True),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    )

        out_numel, out_size = convNoutput([self.conv1, self.conv2, self.conv3], input_size/2)
        # set fc layer based on predicted size
        self.fc = nn.Sequential(
                View(),
                nn.Linear(out_numel, params[3]),
                nn.ReLU()
                )
        self.classifier = classifier = nn.Sequential(
                View(),
                nn.Linear(params[3], 6) # affine transform has 6 parameters
                )
        # initialize stn parameters (affine transform)
        self.classifier[1].weight.data.fill_(0)
        self.classifier[1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def localization_network(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


    def forward(self, x):
        theta = self.localization_network(x)
        theta = theta.view(-1,2,3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
        

class SillNet(nn.Module):
    def __init__(self, nc, input_size, class_train, class_test, extract_chn=None, classify_chn=None, param1=None, param2=None, param3=None, param4=None, param_mask=None):
        super(SillNet, self).__init__()

        self.extract_chn = extract_chn
        self.classify_chn = classify_chn
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        self.param4 = param4
        self.input_size = input_size
        self.nc = nc
        self.class_train = class_train
        self.class_test = class_test

        # extracter
        self.ex_pd1 = nn.ReplicationPad2d(2)
        self.ex1 = nn.Conv2d(nc, self.extract_chn[0], 5, 1) # inchn, outchn, kernel, stride, padding, dilation, groups
        self.ex_bn1 = nn.InstanceNorm2d(self.extract_chn[0])

        self.ex_pd2 = nn.ReplicationPad2d(2)
        self.ex2 = nn.Conv2d(self.extract_chn[0], self.extract_chn[1], 5, 1) # 1/1
        self.ex_bn2 = nn.InstanceNorm2d(self.extract_chn[1])

        self.ex_pd3 = nn.ReplicationPad2d(1)
        self.ex3 = nn.Conv2d(self.extract_chn[1], self.extract_chn[2], 3, 1) # 1/1
        self.ex_bn3 = nn.InstanceNorm2d(self.extract_chn[2])
        
        self.ex_pd4 = nn.ReplicationPad2d(1)
        self.ex4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[3], 3, 1) # 1/1
        self.ex_bn4 = nn.InstanceNorm2d(self.extract_chn[3])
        
        self.ex_pd5 = nn.ReplicationPad2d(1)
        self.ex5 = nn.Conv2d(self.extract_chn[3], self.extract_chn[4], 3, 1) # 1/1
        self.ex_bn5 = nn.InstanceNorm2d(self.extract_chn[4])
        
        
        self.ex_pd6 = nn.ReplicationPad2d(1)
        self.ex6 = nn.Conv2d(self.extract_chn[4], self.extract_chn[5], 3, 1) # 1/1
        self.ex_bn6 = nn.InstanceNorm2d(self.extract_chn[5])
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # decoder
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.de_pd1 = nn.ReplicationPad2d(1)
        self.de1 = nn.Conv2d(int(self.extract_chn[5]/2), self.extract_chn[4], 3, 1)
        self.de_bn1 = nn.InstanceNorm2d(self.extract_chn[4], 1.e-3)

        self.de_pd2 = nn.ReplicationPad2d(1)
        self.de2 = nn.Conv2d(self.extract_chn[4], self.extract_chn[3], 3, 1)
        self.de_bn2 = nn.InstanceNorm2d(self.extract_chn[3], 1.e-3)
        
        self.de_pd3 = nn.ReplicationPad2d(1)
        self.de3 = nn.Conv2d(self.extract_chn[3], self.extract_chn[2], 3, 1)
        self.de_bn3 = nn.InstanceNorm2d(self.extract_chn[2], 1.e-3)
        
        self.de_pd4 = nn.ReplicationPad2d(1)
        self.de4 = nn.Conv2d(self.extract_chn[2], self.extract_chn[1], 3, 1)
        self.de_bn4 = nn.InstanceNorm2d(self.extract_chn[1], 1.e-3)

        self.de_pd5 = nn.ReplicationPad2d(1)
        self.de5 = nn.Conv2d(self.extract_chn[1], nc, 3, 1)
        
        # warping
        if param1 is not None:
            self.stn1 = stn(nc, self.input_size, param1)
        if param2 is not None:
            self.stn2 = stn(self.extract_chn[1], self.input_size, param2)
        if param3 is not None:
            self.stn3 = stn(self.extract_chn[3], self.input_size, param3)
        if param4 is not None:
            self.stn4 = stn(int(self.extract_chn[5]/2), self.input_size, param4)

        # classifier 1
        self.cls1 = nn.Conv2d(int(self.extract_chn[5]), self.classify_chn[0], 5, 1, 2) # inchn, outchn, kernel, stride, padding, dilation, groups                                
        self.cls_bn1 = nn.BatchNorm2d(self.classify_chn[0])

        self.cls2 = nn.Conv2d(self.classify_chn[0], self.classify_chn[1], 5, 1, 2) # 1/2
        self.cls_bn2 = nn.BatchNorm2d(self.classify_chn[1])

        self.cls3 = nn.Conv2d(self.classify_chn[1], self.classify_chn[2], 5, 1, 2) # 1/4
        self.cls_bn3 = nn.BatchNorm2d(self.classify_chn[2])
        
        
        self.cls4 = nn.Conv2d(self.classify_chn[2], self.classify_chn[3], 3, 1, 1) # 1/4
        self.cls_bn4 = nn.BatchNorm2d(self.classify_chn[3])
        
        self.cls5 = nn.Conv2d(self.classify_chn[3], self.classify_chn[4], 3, 1, 1) # 1/8
        self.cls_bn5 = nn.BatchNorm2d(self.classify_chn[4])
        
        self.cls6 = nn.Conv2d(self.classify_chn[4], self.classify_chn[5], 3, 1, 1) # 1/8
        self.cls_bn6 = nn.BatchNorm2d(self.classify_chn[5])
        
        self.fc1 = nn.Linear(int(self.input_size/8*self.input_size/8)*self.classify_chn[5], self.class_train)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        
        # classifier 2
        self.cls21 = nn.Conv2d(int(self.extract_chn[5]), self.classify_chn[0], 5, 1, 2) # inchn, outchn, kernel, stride, padding, dilation, groups                                
        self.cls2_bn1 = nn.BatchNorm2d(self.classify_chn[0])

        self.cls22 = nn.Conv2d(self.classify_chn[0], self.classify_chn[1], 5, 1, 2) # 1/2
        self.cls2_bn2 = nn.BatchNorm2d(self.classify_chn[1])

        self.cls23 = nn.Conv2d(self.classify_chn[1], self.classify_chn[2], 5, 1, 2) # 1/4
        self.cls2_bn3 = nn.BatchNorm2d(self.classify_chn[2])
        
        
        self.cls24 = nn.Conv2d(self.classify_chn[2], self.classify_chn[3], 3, 1, 1) # 1/4
        self.cls2_bn4 = nn.BatchNorm2d(self.classify_chn[3])
        
        self.cls25 = nn.Conv2d(self.classify_chn[3], self.classify_chn[4], 3, 1, 1) # 1/8
        self.cls2_bn5 = nn.BatchNorm2d(self.classify_chn[4])
        
        self.cls26 = nn.Conv2d(self.classify_chn[4], self.classify_chn[5], 3, 1, 1) # 1/8
        self.cls2_bn6 = nn.BatchNorm2d(self.classify_chn[5])       
        
        self.fc2 = nn.Linear(int(self.input_size/8*self.input_size/8)*self.classify_chn[5], self.class_test)
        

    def extract(self, x, is_warping):
        if is_warping and self.param1 is not None:
            x = self.stn1(x)
        h1 = self.leakyrelu(self.ex_bn1(self.ex1(self.ex_pd1(x))))
        h2 = self.leakyrelu(self.ex_bn2(self.ex2(self.ex_pd2(h1))))
        
        if is_warping and self.param2 is not None:
            h2 = self.stn2(h2)
        h3 = self.leakyrelu(self.ex_bn3(self.ex3(self.ex_pd3(h2))))
        h4 = self.leakyrelu(self.ex_bn4(self.ex4(self.ex_pd4(h3))))
        
        if is_warping and self.param3 is not None:
            h4 = self.stn3(h4)
        h5 = self.leakyrelu(self.ex_bn5(self.ex5(self.ex_pd5(h4))))
        h6 = self.sigmoid(self.ex_bn6(self.ex6(self.ex_pd6(h5))))
        
        feat_sem, feat_illu = torch.chunk(h6, 2, 1)
        feat_sem_nowarp = feat_sem
            
        if is_warping and self.param4 is not None:
            feat_sem = self.stn4(feat_sem)
        
        
        return feat_sem, feat_illu, feat_sem_nowarp

    def decode(self, x):
        h1 = self.leakyrelu(self.de_bn1(self.de1(self.de_pd1(x))))
        h2 = self.leakyrelu(self.de_bn2(self.de2(self.de_pd2(h1))))
        h3 = self.leakyrelu(self.de_bn3(self.de3(self.de_pd3(h2))))
        h4 = self.leakyrelu(self.de_bn4(self.de4(self.de_pd4(h3))))
        out = self.sigmoid(self.de5(self.de_pd5(h4)))
        return out
        
    def classify(self, x):
        h1 = self.pool2(self.leakyrelu(self.cls_bn1(self.cls1(x))))
        h2 = self.leakyrelu(self.cls_bn2(self.cls2(h1)))
        h3 = self.pool2(self.leakyrelu(self.cls_bn3(self.cls3(h2))))
        h4 = self.leakyrelu(self.cls_bn4(self.cls4(h3)))
        h5 = self.pool2(self.leakyrelu(self.cls_bn5(self.cls5(h4))))
        h6 = self.leakyrelu(self.cls_bn6(self.cls6(h5)))
        h7 = h6.view(-1, int(self.input_size/8*self.input_size/8*self.classify_chn[5]))
        out = self.fc1(h7)
        return out
    
    def classify2(self, x):
        h1 = self.pool2(self.leakyrelu(self.cls2_bn1(self.cls21(x))))
        h2 = self.leakyrelu(self.cls2_bn2(self.cls22(h1)))
        h3 = self.pool2(self.leakyrelu(self.cls2_bn3(self.cls23(h2))))
        h4 = self.leakyrelu(self.cls2_bn4(self.cls24(h3)))
        h5 = self.pool2(self.leakyrelu(self.cls2_bn5(self.cls25(h4))))
        h6 = self.leakyrelu(self.cls2_bn6(self.cls26(h5)))
        h7 = h6.view(-1, int(self.input_size/8*self.input_size/8*self.classify_chn[5]))
        out = self.fc2(h7)
        return out

    def init_params(self, net):
        print('Loading the model from the file...')
        net_dict = self.state_dict()
        if isinstance(net, dict):
            pre_dict = net
        else:
            pre_dict = net.state_dict()
        # 1. filter out unnecessary keys
        pre_dict = {k: v for k, v in pre_dict.items() if (k in net_dict)}
        net_dict.update(pre_dict)
        # 3. load the new state dict
        self.load_state_dict(net_dict)