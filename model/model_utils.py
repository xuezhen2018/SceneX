import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import fcn

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size+1)//2
    if kernel_size%2 == 1:
        center = factor-1
    else:
        center = factor-0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class FCN8s(nn.Module):
    
    pretrained_model = '../pretrain/fcn8s_from_caffe.pth'
    @classmethod
    def download(cls):
        return fcn.data.cached_download(
                url='http://drive.google.com/uc?id=0B9P1L--7Wd2vT0FtdThWREhjNkU',
                path=cls.pretrained_model,
                md5='dbd9bbb3829a3184913bccc74373afbb',
            )
    
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        
        self._initialize_weight()
        
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
    
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        
        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h
        
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h
        
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        
        h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        
        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h
        
        h = self.score_pool4(pool4)
        h = h[:, :, 5:5+upscore2.size()[2], 5:5+upscore2.size()[3]]
        score_pool4c = h
        
        h = upscore2 + score_pool4c
        h = self.upscore_pool4(h)
        upscore_pool4 = h
        
        h = self.score_pool3(pool3)
        h = h[:, :, 9:9+upscore_pool4.size()[2], 9:9+upscore_pool4.size()[3]]
        score_pool3c = h
        
        h = upscore_pool4 + score_pool3c
        h = self.upscore8(h)
        h = h[:, :, 31:31+x.size()[2], 31:31+x.size()[3]].contiguous()
        
        return h
    
    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)
        
class ResnetBlock(nn.Module):
    def __init__(self, feat_dim, norm_type=1, bias=True, relu_type=1):
        super(ResnetBlock, self).__init__()
        
        self.bias = bias
        self.conv_layer = nn.Conv2d(feat_dim, feat_dim, 3, 1, 1, bias=self.bias)
        self.dropout = nn.Dropout(0.5)
        
        if relu_type==1:
            self.activation = nn.ReLU(True)
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        if norm_type==1:
            self.normalization_layer = nn.BatchNorm2d(feat_dim)
        else:
            self.normalization_layer = nn.InstanceNorm2d(feat_dim)
    
    def forward(self, x):
        response = self.dropout(self.activation(self.normalization_layer(self.conv_layer(x))))
        out = x + response
        return out

class DilatedResnetBlock(nn.Module):
    def __init__(self, feat_dim, norm_type=1, bias=True, relu_type=1):
        super(DilatedResnetBlock, self).__init__()
        
        self.bias = bias
        self.conv_layer = nn.Conv2d(feat_dim, feat_dim, 3, 1, 2, dilation=2, bias=self.bias)
        
        if relu_type == 1:
            self.activation = nn.ReLU(True)
        else:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        if norm_type == 1:
            self.normalization_layer = nn.BatchNorm2d(feat_dim)
        else:
            self.normalization_layer = nn.InstanceNorm2d(feat_dim)
    
    def forward(self, x):
        response = self.activation(self.normalization_layer(self.conv_layer(x)))
        out = x + response
        return out

if __name__ == "__main__":
    net = FCN8s()
    inp = torch.rand(1,3,64,64)
    out = net(inp)
    print(out.shape)