import torch
from tqdm import tqdm

from dataset import get_dataloader
from model import InputMethodModel
import config


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
    训练一个epoch
    :param model:模型
    :param dataloader:数据加载其
    :param loss_function:损失函数
    :param optimizer:优化器
    :param device:设备
    :return:每个batch的平均loss
    """
    epoch_total_loss = 0
    model.train()
    for inputs,targets in tqdm(dataloader,desc="训练"):
        # inputs.shpe:[batch_size, seq_len]
        # targets.shpe:[batch_size]
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)  # output.shpe:[batch_size, vocab_size]
        loss = loss_function(output,targets)
        loss.backward()
        optimizer.step()
        epoch_total_loss+=loss.item()
    return epoch_total_loss/len(dataloader)

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
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

    # 开始训练
    for epoch in range(1,config.EPOCHS+1):
        print(f"======== Epoch {epoch} ========")
        # 训练一轮的逻辑
        avg_loss = train_one_epoch(model,dataloader,loss_function,optimizer,device)
        print(f"Loss:{avg_loss}")

    model

if __name__ == '__main__':
    train()