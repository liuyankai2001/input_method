import jieba
from tqdm import tqdm

from src import config


class JiebaTokenizer:
    unk_token = "<UNK>"
    def __init__(self,vocab_list):
        self.vocab_list = vocab_list
        self.vocab_size = len(vocab_list)
        self.word2index = {word:index for index,word in enumerate(vocab_list)}
        self.index2word = {index:word for index,word in enumerate(vocab_list)}
        self.unk_token_id = self.word2index[self.unk_token]

    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    def encode(self,text):
        word_list = self.tokenize(text)
        return [self.word2index.get(word,self.unk_token_id) for word in word_list]

    @classmethod
    def from_vocab(cls,vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line[:-1] for line in f.readlines()]
        print("词表加载完成")
        return cls(vocab_list)

    @classmethod
    def built_vocab(cls,sentences,vocab_file):
        vocab_set = set()
        for sentence in tqdm(sentences, desc='构建词表'):
            for word in jieba.lcut(sentence):
                vocab_set.add(word)
        vocab_list = [cls.unk_token] + list(vocab_set)
        print(f'词表大小：{len(vocab_list)}')
        word2index = {word: index for index, word in enumerate(vocab_list)}
        # 保存词表
        with open(vocab_file, mode='w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

        print("词表保存完成")

if __name__ == '__main__':
    tokenize = JiebaTokenizer.from_vocab(config.PROCESS_DATA_DIR / 'vocab.txt')
    text = '我觉得你说的很对，但是我不认可,你再说一句我就不理你了'
    print(JiebaTokenizer.tokenize(text))
    index_list = tokenize.encode(text)
    print(index_list)