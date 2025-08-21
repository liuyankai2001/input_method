from torch import nn
import config
class InputMethodModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=config.EMBEDDING_DIM)
        self.rnn = nn.RNN(input_size=config.EMBEDDING_DIM,hidden_size=config.HIDDEN_SIZE,batch_first=True)
        self.linear = nn.Linear(in_features=config.HIDDEN_SIZE,out_features=vocab_size)

    def forward(self,x):
        # x.shape:[batch_size,seq_len]
        embed = self.embedding(x)
        # embed.shape:[batch_size, seq_len, embeding_dim]

        output,hidden = self.rnn(embed)
        # output.shape:[batch_size, seq_len, hidden_dim]
        last_hidden = output[:,-1,:]
        # last_hidden.shape:[batch_size, hidden_dim]
        output = self.linear(last_hidden)
        # output.shape:[batch_size, vocab_size]
        return output