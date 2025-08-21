import torch
from dataset import get_dataloader
from model import InputMethodModel
import config
def train():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备：{device}')
    dataloader = get_dataloader()
    print("数据集加载完成")

    # 加载词表
    with open(config.PROCESS_DATA_DIR / 'vocab.txt','r',encoding='utf-8') as f:
        vocab_list = [line[:-1] for line in f.readlines()]
    print("词表加载完成")
    model = InputMethodModel(vocab_size=len(vocab_list))
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    # 开始训练


if __name__ == '__main__':
    train()