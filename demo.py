from hcts import Ts_Train, Ts_Dataloader
from torch import nn
import torch


class lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=128, num_layers=3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 1)

    def forward(self, X):
        out, _ = self.lstm(X)
        out = self.fc(out[:, -1, :])
        return out

def tras(df):
    return (df - df.mean())/(df.std()+1e-6)

def prework(model):
    state_dict = torch.load('lstm_50epochs.pth')
    model.load_state_dict(state_dict)
    return model

td = Ts_Dataloader('data/train_data.csv', input_time_steps=7, output_time_steps=1, output_idx=0, batch_size=128,
                   shuffle=True, transform=tras)
vd = Ts_Dataloader('data/test_data.csv',input_time_steps=7,output_time_steps=1,output_idx=0,batch_size=128,
                   shuffle=True, transform=tras)
model = lstm()
Ts_Train(model=model, learning_rate=1e-3, num_epochs=50, train_dataloader=td,
         validation_dataloader=vd, save_results=True, log_process=True, pre_work=prework)
