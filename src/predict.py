import jieba
import torch
import config
from model import InputMethodModel


def predict_batch(model,input_tensor):
    """
    批量预测
    :param model:模型
    :param input_tensor:输入张量 [batch_size, sql_len]
    :return:[[1,2,3,4,5],[6,7,8,9,10]]
    """
    model.eval()
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)  # [batch_size, vocab_size]
        top5_indexes = torch.topk(output, dim=1, k=5).indices
        # top5 [batch_size, 5]
    top5_indexes_list = top5_indexes.tolist()
    return top5_indexes_list
def predict(text,model,word2index,index2word,device):

    # 数据
    word_list = jieba.lcut(text)
    index_list = [word2index.get(word,0) for word in word_list]
    input_tensor = torch.tensor([index_list]).to(device) #[batch_size, seq_len]

    top5_indexes_list = predict_batch(model,input_tensor)

    top5_words = [index2word.get(index) for index in top5_indexes_list[0]]
    return top5_words

def run_predict():
    # 加载资源
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(config.PROCESS_DATA_DIR / 'vocab.txt','r',encoding='utf-8') as f:
        vocab_list = [line[:-1] for line in f.readlines()]
    word2index = {word:index for index, word in enumerate(vocab_list)}
    index2word = {index:word for index, word in enumerate(vocab_list)}
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR/'model.pt'))


    history_input = ''
    print("请输入下一个词：（输入q或quit退出）")
    while True:
        user_input = input("> ")
        if user_input in ['q','quit']:
            print("程序已退出")
            break
        if user_input == "":
            print("请输入下一个词")
            continue
        history_input+=user_input
        top5_words = predict(history_input,model,word2index,index2word,device)
        print(f"历史输入：{history_input}")
        print(top5_words)



if __name__ == '__main__':
    # top5_words = predict("我们团队正在")
    # print(top5_words)
    run_predict()