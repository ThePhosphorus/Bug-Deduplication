
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMEncoder(nn.Module):
    def __init__(self, embed_dim: int = 50, hid_dim: int = 200, dropout: float = 0.1) -> None:
        super().__init__()
        self.hid_dim: int = hid_dim
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=self.hid_dim//2, bidirectional=True, num_layers=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb_f: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
        out, _ = self.lstm(emb_f)
        #return torch.nanmean(out, dim=1)
        x, lengths =  torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        #mask = torch.arange(x.shape[1], device=lengths.device).expand(x.shape[0], x.shape[1]) < lengths.unsqueeze(1)
        lengths = lengths - 1
        result = x.take_along_dim(lengths.to(device=x.device)[:,None,None], dim=1)
        #mean, _ = torch.max(x, dim= 1)
        #return mean
        return torch.cat((result[:,0,:self.hid_dim//2], x[:,0,self.hid_dim//2:]), dim=-1) # returns shape (B, hidden_dim)

class Classifier(nn.Module):
    def __init__(self, input_dim: int = 50, out_num: int = 2, dropout: float = 0.1, features_num: int = 4) -> None:
        super().__init__()
        self.features_num = features_num
        self.input_dim = self.features_num * input_dim
        self.out_num = out_num
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(in_features= self.input_dim,out_features=self.input_dim//2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=self.input_dim//2,out_features= self.out_num)

    def forward(self, v1: torch.tensor, v2: torch.tensor) -> torch.Tensor :
        diff = torch.abs(v1 - v2) # B, H
        if self.features_num == 1:
            features = diff # B, H
        elif self.features_num == 2:
            features = torch.cat((diff, (v1 + v2) / 2), -1) # B, 2H
        elif self.features_num == 3:
            features = torch.cat((diff, (v1 + v2) / 2, v1 * v2), -1) # B, 3H
        elif self.features_num == 4:
            features = torch.cat((diff, v1, v2, v1 * v2), -1) # B, 4H
        elif self.features_num == 5:
            features = torch.cat((diff, v1, v2, (v1 + v2) / 2, v1 * v2), -1) # B, 5H
        else:
            raise ValueError("Wrong features_num parameter value")
        features = self.dropout(features) # B, FH
        features = self.relu(self.linear1(features)) # B, FH/2
        cl = self.linear2(features) # B, O
        if self.out_num == 1:
            cl = torch.cat((1 - cl, cl), dim=-1)
        else:
            return cl

class S3M(nn.Module) :
    def __init__(self, embed_dim: int = 50, hid_dim: int= 200, dropout: float = 0.1, feature_number: int = 4, use_classifier: bool = True) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(embed_dim=embed_dim, hid_dim=hid_dim, dropout=dropout)
        self.classifier = Classifier(input_dim=hid_dim, dropout=dropout, features_num= feature_number) if use_classifier else self._shoud_not_be_called

    def forward(self, stack1: torch.nn.utils.rnn.PackedSequence, stack2: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
        return self.classifier(self.encoder(stack1), self.encoder(stack2))

    def _shoud_not_be_called(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
                 
