import torch
import config
from dataset import get_dataloader
from tokensizer import ChineseTokenizer,EnglishTokenizer
from model import TranslationModel
from predict import predict_batch
from nltk.translate.bleu_score import corpus_bleu
def evaluate(device,model,test_dataloader,en_tokenizer):
    predictions = []
    # predictions = [[*,*,*,],[*,*,*,*,*,*,*],[*,*,*,*],[*,*,*,*,*,*,]]
    references = []
    # references = [[[参考译文1,参考译文2]],[[参考译文1,参考译文2参考译文1,参考译文3]]]
    for inputs,targets in test_dataloader:
        inputs = inputs.to(device)
        # inputs.shape:[batch_size,seq_len]
        targets = targets.tolist()
        #targets.shape:[[*,*,*,*,*,*,pad],[*,*,*,*,*,*,*],[*,*,*,*,*,pad,pad],[*,*,*,*,*,*,*]]
        batch_result = predict_batch(model,inputs,en_tokenizer) # eg:[[],[],[]]
        predictions.extend(batch_result) # append会变成3维
        references.extend([[target[1:target.index(en_tokenizer.eos_token_index)]] for target in targets])
    return corpus_bleu(references,predictions)

def run_evaluate():
    # 1.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2.词表
    zh_tokenizer = ChineseTokenizer.from_voacb(config.MODELS_DIR/'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_voacb(config.MODELS_DIR/'en_vocab.txt')
    # 3.模型
    model = TranslationModel(zh_tokenizer.vocab_size,en_tokenizer.vocab_size,
                             zh_tokenizer.pad_token_index,en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'best.pt'))

    # 4.评估数据集
    test_dataloader = get_dataloader(train=False)

    # 5.评估逻辑
    bleu = evaluate(device,model,test_dataloader,en_tokenizer)
    print(f"评估结果：bleu:{bleu}")

if __name__ == '__main__':
    run_evaluate()