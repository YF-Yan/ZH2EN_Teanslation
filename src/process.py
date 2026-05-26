import pandas as pd
from sklearn.model_selection import train_test_split
from tokensizer import ChineseTokenizer,EnglishTokenizer
import config

def process():
    print("开始处理数据...")
    df = pd.read_csv(config.RAW_DATA_DIR/'cmn.txt',
                sep='\t', # 分隔符类型
                header=None, # 没表头
                usecols=[0,1],# 取0，1列
                names=['en','zh'],# 给个名字
                encoding='utf-8').dropna()

    train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)

    # 构建词表
    ChineseTokenizer.build_vocab(train_df['zh'].tolist(),config.MODELS_DIR/'zh_vocab.txt')
    EnglishTokenizer.build_vocab(train_df['en'].tolist(),config.MODELS_DIR/'en_vocab.txt')

    # 构建tokenizer
    zh_tokenizer = ChineseTokenizer.from_voacb(config.MODELS_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_voacb(config.MODELS_DIR / 'en_vocab.txt')

    # 构建训练集
    train_df['zh'] = train_df['zh'].apply(lambda x:zh_tokenizer.encode(x))
    train_df['en'] = train_df['en'].apply(lambda x:en_tokenizer.encode(x,add_sos_eos=True))

    train_df.to_json(config.PROCESS_DATA_DIR / 'train.jsonl',orient='records',lines=True)

    test_df['zh'] = test_df['zh'].apply(lambda x: zh_tokenizer.encode(x))
    test_df['en'] = test_df['en'].apply(lambda x: en_tokenizer.encode(x, add_sos_eos=True))

    test_df.to_json(config.PROCESS_DATA_DIR / 'test.jsonl', orient='records', lines=True)
    print("数据处理完毕。")


if __name__ == '__main__':
    process()