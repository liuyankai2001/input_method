import torch

from src import config
from src.model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch


def evaluate_model(model, dataloader, device):
    total_count = 0
    top1_acc_count = 0
    top5_acc_count = 0
    for inputs,targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.tolist()  #[batch_size]
        top5_index_list = predict_batch(model,inputs)

        for target,top5_indexes in zip(targets,top5_index_list):
            total_count+=1
            if target==top5_indexes[0]:
                top1_acc_count+=1
            if target in top5_indexes:
                top5_acc_count+=1
    return top1_acc_count/total_count,top5_acc_count/total_count
def run_evaluate():
    # 加载资源
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(config.PROCESS_DATA_DIR / 'vocab.txt', 'r', encoding='utf-8') as f:
        vocab_list = [line[:-1] for line in f.readlines()]
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    dataloader = get_dataloader(False)

    # 评估模型
    top1_acc,top5_acc = evaluate_model(model,dataloader,device)
    print(f'====== 评估结果 ======')
    print(f'top1准确率：{top1_acc}')
    print(f'top5准确率：{top5_acc}')
    print(f'====================')

if __name__ == '__main__':
    run_evaluate()