# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .ops import Conv_2d, Res_2d
from torch.autograd import Variable



class CPC(nn.Module):
    def __init__(self, K, seq_len):
        super(CPC, self).__init__()

        self.seq_len = seq_len
        self.K = K
        self.c_size = 256
        self.z_size = 512
        self.dwn_fac = 513
        
        self.encoder = nn.Sequential( 
            nn.Conv1d(1, self.z_size, kernel_size=16, stride=8, padding=3, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=8, stride=4, padding=2, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.z_size, self.z_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.z_size),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(self.z_size, self.c_size, num_layers = 1, 
                          bidirectional = False, batch_first = True)
        
        # These are all trained
        self.Wk = nn.ModuleList([nn.Linear(self.c_size,self.z_size) for i in range(self.K)])
        self.softmax = nn.Softmax(dim=1)
        self.lsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        x: torch.float32 shape (batch_size,seq_len)
        hidden: torch.float32 
                shape (num_layers*num_directions=1,batch_size,hidden_size=256) 
        """
        batch_size = x.size()[0] 
        x = x.unsqueeze(1)
        z = self.encoder(x) 
        
        z = z.transpose(1,2)
        highest = self.seq_len//self.dwn_fac - self.K 
        time_C = torch.randint(highest, size=(1,)).long()

        z_t_k = z[:, time_C + 1:time_C + self.K + 1, :].clone().cpu().float()
        z_t_k = z_t_k.transpose(1,0)
        
        z_0_T = z[:,:time_C + 1,:]
        output, hidden = self.gru(z_0_T)
        c_t = output[:,time_C,:].view(batch_size, self.c_size) 
        
        W_c_k = torch.empty((self.K, batch_size, self.z_size)).float() 
        for k in np.arange(0, self.K):
            linear = self.Wk[k] # c_t is size 256, Wk is a 512x256 matrix 
            W_c_k[k] = linear(c_t) # Wk*c_t e.g. size 8*512
            
        nce = 0 # average over timestep and batch

        for k in np.arange(0, self.K):    
            # (batch_size, z_size)x(z_size, batch_size) = (batch_size, batch_size)
            zWc = torch.mm(z_t_k[k], torch.transpose(W_c_k[k],0,1))     
            logsof_zWc = self.lsoftmax(zWc)
            nce += torch.sum(torch.diag(logsof_zWc)) # nce is a tensor

        nce /= -1.*batch_size*self.K
        argmax = torch.argmax(self.softmax(zWc), dim=0)
        correct = torch.sum( torch.eq(argmax, torch.arange(0, batch_size)))
        accuracy = torch.tensor(1.*correct.item()/batch_size)
        return accuracy, nce, hidden

    def predict(self, x):
        x = x.unsqueeze(1)
        z = self.encoder(x) 
        z = z.transpose(1,2) 
        output, hidden = self.gru(z)

        # return output, hidden # return every frame
        return hidden, output[:,-1,:]  # only return the last frame per utt


class FCN05(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50):
        super(FCN05, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # FCN
        self.layer1 = Conv_2d(1, 64, pooling=(4,2))
        self.layer2 = Conv_2d(64, 128, pooling=(2,2))
        self.layer3 = Conv_2d(128, 128, pooling=(2,2))
        self.layer4 = Conv_2d(128, 128, pooling=(2,2))
        self.layer5 = Conv_2d(128, 64, pooling=(3,2))

        # Dense
        self.dense = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spectrogram
        x = x.unsqueeze(1)
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        embedding = x.view(x.size(0), -1)
        # x = self.dropout(embedding)
        # x = self.dense(x)
        # x = self.sigmoid(x)
        # return x, embedding_by_frame
    
    def get_emb(self, x):
        # Spectrogram
        x = x.unsqueeze(1)
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        embedding_by_frame = x.permute(0,2,3,1).squeeze(1)
        return embedding_by_frame

class FCN(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50):
        super(FCN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # FCN
        self.layer1 = Conv_2d(1, 64, pooling=(2,4))
        self.layer2 = Conv_2d(64, 128, pooling=(2,4))
        self.layer3 = Conv_2d(128, 128, pooling=(2,4))
        self.layer4 = Conv_2d(128, 128, pooling=(3,5))
        self.layer5 = Conv_2d(128, 64, pooling=(4,4))

        # Dense
        self.dense = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        embedding = x.view(x.size(0), -1)
        x = self.dropout(embedding)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x, embedding

    def get_emb(self, x):
        # Spectrogram
        x = x.unsqueeze(1)
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)
        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        embedding_by_frame = x.permute(0,2,3,1).squeeze(1)
        return embedding_by_frame
class ShortChunkCNN_Res(nn.Module):
    '''
    Short-chunk CNN architecture with residual connections.
    '''
    def __init__(self,
                n_channels = 128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50
                ):
        super(ShortChunkCNN_Res, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        embedding = x.squeeze(2)

        # Dense
        x = self.dense1(embedding)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x, embedding

    def get_emb(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)

        return x.squeeze(2)