import jieba
import torch
import config
from tokensizer import ChineseTokenizer,EnglishTokenizer
from model import TranslationModel

def predict_batch(model,inputs,en_tokenizer):
    """

    :param model:
    :param inputs: 输入 shape:{batch_size,seq_len}
    :return: shape:{[*,*,eos],[*,*,*,*,*,eos],[],,,]}
    """
    model.eval()
    device = inputs.device
    batch_size = inputs.shape[0]
    # 预测缓存、
    generated = []

    is_finish = torch.full([batch_size],False,device=device)
    with torch.no_grad():
        context_vector = model.encoder(inputs)
        # context_vector : {batch_size,hiddensize}
        decoder_hidden = context_vector.unsqueeze(0)
        # decoder_hidden :{1,batch_size,hiddensize}
        decoder_input = torch.full([batch_size,1],en_tokenizer.sos_token_index,device=device)
        # decoder_input : {batch_size,1}

        for i in range(config.MAX_SEQ_LENGTH):
            # 解码
            decoder_output,decoder_hidden = model.decoder(decoder_input,decoder_hidden)
            # decoder_output : {batch_size,1,vocab_size}

            # 保存预测结果
            next_token_indexes = torch.argmax(decoder_output,dim=-1)
            # next_token_indexes : {batch_size,1}
            generated.append(next_token_indexes)

            # 更新输入
            decoder_input = next_token_indexes

            # 判断是否结束
            is_finish |= next_token_indexes.squeeze(1) == en_tokenizer.eos_token_index
            if is_finish.all():
                break


        # 处理预测结果
        # generated : {tensor[batch_size,1]}
        generated_tensor = torch.cat(generated,dim=1)
        # generated_tensor : {batch_size,seq_len}
        generated_list = generated_tensor.tolist() # {[*,*,eos,*,*,*],[*,*,*,*,*,eos],[],,,]}

        # 去eos后token_id
        for index,sentence in enumerate(generated_list):
            if en_tokenizer.eos_token_index in sentence:
                eos_pos = sentence.index(en_tokenizer.eos_token_index)
                generated_list[index] = sentence[:eos_pos]

        return generated_list

def predeict(text,device,model,zh_tokenizer,en_tokenizer):

    # 1.处理输入,注意，前向传播函数的输入x是序列id，在函数内有embed词嵌入
    indexes = zh_tokenizer.encode(text)
    input_tensor = torch.tensor([indexes],dtype=torch.long)
    input_tensor = input_tensor.to(device)
    # 2.预测
    batch_result = predict_batch(model,input_tensor,en_tokenizer)

    return en_tokenizer.decode(batch_result[0])


def run_predict():
    # 1.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2.词表
    zh_tokenizer = ChineseTokenizer.from_voacb(config.MODELS_DIR/'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_voacb(config.MODELS_DIR/'en_vocab.txt')
    # 3.模型
    model = TranslationModel(zh_tokenizer.vocab_size,en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR /'best.pt'))

    print("欢迎使用中英翻译")
    while True :
        user_input = input("中文>")
        if user_input in ['q','quit']:
            break
        if user_input.strip() == '':
            continue
        # print(user_input)
        result = predeict(user_input,device,model,zh_tokenizer,en_tokenizer)
        print(f"译文：{result}")



if __name__ == '__main__':
    run_predict()