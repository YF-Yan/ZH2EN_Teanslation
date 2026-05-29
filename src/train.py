import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import get_dataloader
from tokensizer import ChineseTokenizer,EnglishTokenizer
from model import TranslationModel


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for inputs,targets in tqdm(dataloader,desc='训练'):
        encoder_inputs = inputs.to(device) #[batch_size,src_seq_len]
        targets = targets.to(device) #[batch_size,tgt_seq_len]
        decoder_inputs = targets[:,:-1]
        decoder_targets = targets[:,1:]
        # 前向
        encoder_outputs,context_vector = model.encoder(encoder_inputs)
        # context_vector : [batch_size,hidden_size]
        # 他与解码输入的形状略不同

        # 解码
        decoder_outputs = []
        decoder_hidden = context_vector.unsqueeze(0)
        seq_len = decoder_inputs.shape[1]
        for i in range(seq_len):
            decoder_input = decoder_inputs[:,i].unsqueeze(1) #[batch_size,1]
            decoder_output,decoder_hidden = model.decoder(decoder_input,decoder_hidden,encoder_outputs)
            # decoder_output :[batch_size,1,vocab_size]
            decoder_outputs.append(decoder_output)

        # decoder_outputs : [tensor(batch_size,1,vocab_size)] --> [batch_size * seqlen,vocab_size]
        # decoder_targets : [batch_size,seqlen] -- >[batch_size * seqlen]
        decoder_outputs = torch.cat(decoder_outputs,dim=1)
        # decoder_outputs : [batch_size,seqlen,vocab_size]
        decoder_outputs = decoder_outputs.reshape(-1,decoder_outputs.shape[-1])
        decoder_targets = decoder_targets.reshape(-1)
        loss = loss_fn(decoder_outputs,decoder_targets)
        # 反向
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss


    return total_loss / len(dataloader)



def train():
    # 1.设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2.数据
    dataloader = get_dataloader()
    # 3.分词器
    zh_tokenizer=ChineseTokenizer.from_voacb(config.MODELS_DIR/'zh_vocab.txt')
    en_tokenizer=EnglishTokenizer.from_voacb(config.MODELS_DIR/'en_vocab.txt')
    # 4.模型
    model = TranslationModel(zh_tokenizer.vocab_size,en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,en_tokenizer.pad_token_index).to(device)
    # 5.损失函数
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_index)
    # 6.优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)
    # 7.TensorBoard
    writer = SummaryWriter(log_dir=config.LOGS_DIR/time.strftime('%Y-%m-%d-%H-%M-%S'))

    best_loss = float('inf')
    for epoch in range(1,config.EPOCHS):
        print(f'------------Epoch{epoch}------------')
        loss = train_one_epoch(model,dataloader,loss_fn,optimizer,device)
        print(f'Loss:{loss:.4f}')

        #记录到TensorBoard
        writer.add_scalar('Loss',loss,epoch)
        if loss < best_loss :
            best_loss = loss
            torch.save(model.state_dict(),config.MODELS_DIR/'best.pt')
            print('保存模型')
    writer.close()

if __name__ =='__main__':
    train()
