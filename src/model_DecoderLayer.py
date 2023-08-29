
import torch.nn as nn
import torch

class Reconstruct_Decoder_MLP(nn.Module):
    '''
    embedding single time point to feature space
    '''
    def __init__(self, in_dim, out_dim): # xdim->256->128->64
        super().__init__()
        mid_dim = 128
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.bn = nn.BatchNorm1d(mid_dim, eps=0.001)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        """
        :param x: B * L * C
        :return: B * L * X_dim
        """

        B, L, C = x.shape
        x = x.reshape(-1, C)  # B*L*C -> BL * C
        x = self.fc2(self.act(self.bn(self.fc1(x))))
        x = x.reshape(B, L, -1)
        return x
    
class Projector(nn.Module):
    '''
    Projector
    '''
    def __init__(self, in_dim, out_dim): # projector
        super().__init__()
        # first layer
        mid_dim = 256
        self.projector = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.projector(x)
        return x

class Point_Predictor(nn.Module):
    '''
    Projector
    '''
    def __init__(self, in_dim, out_dim): # projector
        super().__init__()
        # first layer
        mid_dim = 128
        self.projector = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim, bias=True))

    def forward(self, x):
        x = self.projector(x)
        return x

class Context_Predictor(nn.Module):
    '''
    Context_Predictor -> attention
    '''
    def __init__(self, embed_dim=256, num_heads=4): # projector
        super().__init__()
        self.context_predictor = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, q, k, v):
        attn_output, attn_output_weights = self.context_predictor(q, k, v)
        return attn_output, attn_output_weights
        attn_output, attn_output_weights = multihead_attn(query, key, value)
        


class Poolinger(nn.Module):
    def __init__(self, kernel_size, stride, mode="avg"): # projector
        super().__init__()
        if mode == "avg":
            self.m = nn.AvgPool1d(kernel_size, stride)
        if mode == "max":
            self.m = nn.MaxPool1d(kernel_size, stride)

    def forward(self, x):
        """Input: B*C*L"""
        x = x.permute(0, 2, 1)
        x = self.m(x)
        x = x.permute(0, 2, 1)
        return x



class Reconstruct_Decoder_RNN(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, out_size, num_layers=1):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(Reconstruct_Decoder_RNN, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_size, eps=0.001)

    def forward(self, x_input, encoder_hidden_states):
        rnn_out, hn = self.rnn(x_input, encoder_hidden_states)
        rnn_out = self.act(rnn_out)
        rnn_out = rnn_out.permute(0, 2, 1)
        rnn_out = self.bn(rnn_out)
        rnn_out = rnn_out.permute(0, 2, 1)
        output = self.linear(rnn_out)
        return output



class Context_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_layers): # xdim->256->128->64
        super().__init__()
        self.decoder = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=rnn_layers, batch_first=True)
    def forward(self, x):
        z = self.decoder(x)
        return z


class Predict_Decoder(nn.Module):
    def __init__(self, in_dim, out_dim): # xdim->256->128->64
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.bn = nn.BatchNorm1d(64, eps=0.001)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.fc2(self.act(self.bn(self.fc1(x))))
        return x


