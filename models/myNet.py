import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import numpy as np
class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidC = int(args.hidCNN)
        self.hidR = int(args.hidRNN)
        self.Ck = int(args.CNN_kernel)
        self.hw = args.highway_window
        self.steps = args.steps
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.feature_num = args.feature_num
        self.poolTimes = args.poolTimes

        
        # self.highway = nn.Linear(self.hw*self.hidC*self.m, self.steps*self.m)
        self.drop = nn.Dropout(args.dropout)

# aspp
        self.aspp = ASPP(in_channel=1,depth=self.hidC,k_size=self.Ck)
        self.se1 = SE_Block(self.hidC)
        self.norm1 = nn.BatchNorm2d(self.hidC)
        self.avepool = nn.AvgPool2d(kernel_size=(3,1),stride=(self.poolTimes,1))
# GUR
        self.GRU = nn.GRU(self.hidC, self.hidR)
# sr        
        self.sr = SRlayer_(1)
        self.srconv = nn.Conv2d(in_channels=1,out_channels=self.hidC,kernel_size=3,padding=1)
        self.se2 = SE_Block(self.hidC)
        self.norm2 = nn.BatchNorm2d(self.hidC)
        self.maxpool = nn.MaxPool2d(kernel_size=(3,1),stride=(self.poolTimes,1))
# transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,nhead=self.nhead)
        self.encoder= nn.TransformerEncoder(self.encoder_layer,num_layers=self.num_layers)
        # self.ts_conf = nn.Conv1d()
# output
        self.atten = FullAttention()
        # self.fulltten = AttentionLayer(self.atten,d_model=(self.P//4)*self.hidC*self.m, n_heads = 8)
        self.fulltten = AttentionLayer(self.atten,32*self.feature_num, n_heads = 8)

        self.last_maxpool = nn.AdaptiveAvgPool2d((1,1))

        self.outFc = nn.Linear((self.P//self.poolTimes)*self.hidC*self.m//self.steps, self.m)
        self.mix_conv = nn.Conv1d(self.P*self.m*2//self.poolTimes,self.P*self.m//self.poolTimes,kernel_size=3,stride=1, padding=1)
        self.bn = nn.BatchNorm1d(self.P*self.m//self.poolTimes)
        # self.active = nn.ELU()

        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh
 
    def forward(self, x):
        # ASPP-CNN
        src = x.view(-1, 1, self.P, self.m)
        c = (self.aspp(src))
        c = self.se1(c)
        if self.poolTimes>1:
            c = self.avepool(c)
        
        # rnn
        r_l = c.view(-1,self.hidC,c.shape[2]*c.shape[3])
        r = r_l.permute(2, 0, 1).contiguous()
        r,_ = self.GRU(r)

        
        # SR
        sr = self.sr(src)
        sr = (self.srconv(sr))
        sr = self.se2(sr)
        if self.poolTimes>1:
            sr = self.maxpool(sr)

        
        # transformer
        sr = sr.view(-1,self.hidC,sr.shape[2]*sr.shape[3])
        sr = sr.permute(2,0,1).contiguous()
        ts = self.encoder(sr)

    
        ts = ts.permute(1,0,2).contiguous()
        # ts = self.bn(ts)
        r = r.permute(1,0,2).contiguous()
        # r = self.bn(r)

        mix = torch.cat([ts,r],dim=1)
        mix = self.mix_conv(mix)
        out = self.bn(mix)
        # out,_ = self.fulltten(mix,mix,mix)
        # out,_ = self.fulltten(ts,ts,r)


        out = out.view(-1,self.steps,(out.shape[1]*self.hidC)//self.steps)
        out = self.outFc(out)

        out = out.view(-1,self.m)

        return out

class ASPP(nn.Module):
    def __init__(self, in_channel=1, depth=100, k_size = 3):
        super(ASPP,self).__init__()
        #F = 2*（dilation -1 ）*(kernal -1) + kernal

        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, k_size, 1, padding=2, dilation=2)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, k_size, 1, padding=4, dilation=4)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, k_size, 1, padding=8, dilation=8)
        self.atrous_block24 = nn.Conv2d(in_channel, depth, k_size, 1, padding=16, dilation=16)
        # self.atrous_block30 = nn.Conv2d(in_channel, depth, k_size, 1, padding=10, dilation=10)
        # self.atrous_block36 = nn.Conv2d(in_channel, depth, k_size, 1, padding=12, dilation=12)
        # self.atrous_block42 = nn.Conv2d(in_channel, depth, k_size, 1, padding=14, dilation=14)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        atrous_block24 = self.atrous_block24(x)
        # atrous_block30 = self.atrous_block30(x)
        # atrous_block36 = self.atrous_block36(x)
        # atrous_block42 = self.atrous_block42(x)
 
        net = self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18,atrous_block24], dim=1))
        return net   
    
# class SRlayer_(nn.Module):
#     def __init__(self,channel):
#         super(SRlayer_,self).__init__()
#         self.channel = channel
#         self.amp_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1,padding=1)
#         self.amp_bn  = nn.BatchNorm2d(channel)
#         self.amp_relu = nn.ReLU()
#         self.phase_conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1,padding=1)
#         self.Relu = nn.ReLU()

#     def forward(self,x):
#         rfft = torch.fft.rfftn(x)
#         amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
#         log_amp = torch.log(amp)
#         phase = torch.angle(rfft)
#         amp_filter = self.amp_conv(log_amp)
#         amp_sr = log_amp - amp_filter
#         SR = torch.fft.irfftn((amp_sr+1j*phase),x.size())
#         SR = self.amp_bn(SR)
#         # amp_sr = self.amp_relu(SR)
#         # SR = self.Relu(SR)
#         return x + SR
class SRlayer_(nn.Module):
    def __init__(self,channel):
        super(SRlayer_,self).__init__()
        self.channel = channel
        self.batch = 1
        self.output_conv = nn.Conv2d(2,3, kernel_size=1)
        self.bn  = nn.BatchNorm2d(3)
        self.Relu = nn.ReLU()
        
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out', nonlinearity='relu')
        
        self.kernalsize = 3
        self.amp_conv = nn.Conv2d(self.channel, self.channel, kernel_size=self.kernalsize, stride=1,padding=1, bias=False)
        self.fucker = np.zeros([self.kernalsize,self.kernalsize])
        for i in range(self.kernalsize):
            for j in range(self.kernalsize):
                self.fucker[i][j] = 1/np.square(self.kernalsize)
        self.aveKernal = torch.Tensor(self.fucker).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.amp_conv.weight = nn.Parameter(self.aveKernal, requires_grad=False)
        
        self.amp_relu = nn.ReLU()
        self.gaussi = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1,padding=1, bias=False)
        self.gauKernal = torch.Tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).unsqueeze(0).unsqueeze(0).repeat(self.batch,self.channel,1,1)
        self.gaussi.weight = nn.Parameter(self.gauKernal,requires_grad=False)

    # def forward(self,x):
    #     rfft = torch.fft.fftn(x)
    #     amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
    #     log_amp = torch.log(amp)
    #     phase = torch.angle(rfft)
    #     amp_filter = self.amp_conv(log_amp)
    #     amp_sr = log_amp - amp_filter
    #     SR = torch.fft.ifftn(torch.exp(amp_sr+1j*phase))
    #     SR = torch.abs(SR)
    #     SR = self.gaussi(SR)
    #     SR = (SR>torch.mean(SR))*torch.FloatTensor(1).cuda()
    #     SR = torch.cuda.FloatTensor(SR)
    #     return SR        
    def forward(self,x):
        out = []
        for batch in range(x.shape[0]):
            rfft = torch.fft.fftn(x[batch].unsqueeze(0))
            amp = torch.abs(rfft) + torch.exp(torch.tensor(-10))
            log_amp = torch.log(amp)
            phase = torch.angle(rfft)
            amp_filter = self.amp_conv(log_amp)
            amp_sr = log_amp - amp_filter
            SR = torch.fft.ifftn(torch.exp(amp_sr+1j*phase))
            SR = torch.abs(SR)
            SR = self.gaussi(SR)
            out.append(SR)
            # SR = (SR>SR.mean())
        return torch.cat(out,dim=0)
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=4):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # print(queries.shape)
        # print(keys.shape)
        # print(values.shape)
        # print(scores.shape)
        # print(V.shape)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

