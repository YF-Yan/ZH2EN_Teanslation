import torch
from torch import nn

from src import config


class TranslationEncoder(nn.Module):
    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

    def forward(self,x):
        # x : [batch_size,seq_len]
        embed = self.embedding(x)
        # embed : [batch_size,seq_len,embedding_dim]
        output,hidden = self.gru(embed)
        # output : [batch_size,seq_len,hidden_size]
        # 输出的seqlen是每个token对应的输出状态，我们需要最后一个有效token的输出状态
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = output[torch.arange(output.shape[0]),lengths - 1]
        # last_hidden_state : [batch_size,hidden_size]
        return last_hidden_state

class TranslationDecoder(nn.Module):
    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,
                                out_features=vocab_size)

    def forward(self,x,hidden_0):
        # x:[batch_size,seqlen]
        # hidden : [1,batch_size,hiddensize]
        embed = self.embedding(x)
        # embed : [batch_size,seqlen,embeddingdim]
        output,hidden_n = self.gru(embed,hidden_0)
        # output [batch_size,1,hiddensize]
        output = self.linear(output)
        # output [batch_size,1,vocabsize]
        return output,hidden_n

class TranslationModel(nn.Module):
    def __init__(self,zh_vocab_size,en_vocab_size,zh_padding_index,en_padding_index):
        super().__init__()
        self.encoder = TranslationEncoder(vocab_size=zh_vocab_size,padding_index=zh_padding_index)
        self.decoder = TranslationDecoder(vocab_size=en_vocab_size,padding_index=en_padding_index)