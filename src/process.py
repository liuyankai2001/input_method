import jieba
import pandas as pd
import json
from pathlib import Path
import logging

from src.tokenizer import JiebaTokenizer

jieba.setLogLevel(logging.ERROR)

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# print(__file__)
import config

def built_dataset(sentences,tokenizer):
    """
    构建数据集
    :param sentences: 原始句子列表['我爱自然语言','我不爱自然语言']
    :param tokenizer:分词器对象
    :return:[{input:[1,2,3,4,5],target:6},{input:[2,3,4,5,6],target:7},...]
    """
    indexed_sentence = [tokenizer.encode(sentence) for sentence in sentences]
    dataset = []  # [{input:[1,2,3,4,5],target:6},{input:[2,3,4,5,6],target:7},...]
    for sentence in indexed_sentence:
        # sentence = [1,2,3,4,5,6,7,8,9,10]
        for i in range(len(sentence) - config.SEQ_LEN):
            input = sentence[i:i + config.SEQ_LEN]
            target = sentence[i + config.SEQ_LEN]
            dataset.append({'input': input, 'target': target})
    return dataset



def process():
    """
    预处理数据
    :return:
    """
    print("开始处理数据")
    df = pd.read_json(config.RAW_DATA_DIR / 'synthesized_.jsonl',orient='records',lines=True).sample(frac=0.1)

    # 抽取句子
    sentences = []
    for dialog in df['dialog']:
        for sentence in dialog:
            sentences.append(sentence.split('：')[1])
    print(f'句子总数：{len(sentences)}')

    # 划分数据集
    train_sentences,test_sentences = train_test_split(sentences,test_size=0.2)
    print(f'训练集数目：{len(train_sentences)}')
    print(f'测试集数目：{len(test_sentences)}')

    JiebaTokenizer.built_vocab(sentences,config.PROCESS_DATA_DIR/'vocab.txt')

    tokenizer = JiebaTokenizer.from_vocab(config.PROCESS_DATA_DIR/'vocab.txt')

    # 构建训练集并保存
    train_dataset = built_dataset(train_sentences,tokenizer)

    pd.DataFrame(train_dataset).to_json(config.PROCESS_DATA_DIR / 'index.train.jsonl',lines=True,orient='records')
    # 构建测试集并保存
    test_dataset = built_dataset(test_sentences, tokenizer)

    pd.DataFrame(test_dataset).to_json(config.PROCESS_DATA_DIR / 'index.test.jsonl', lines=True, orient='records')


    print("数据处理完成")


if __name__ == '__main__':
    process()