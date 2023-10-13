import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def Ts_Dataloader(path, input_time_steps,  output_time_steps, input_idx=None, output_idx=None, transform=None, batch_size=1, shuffle=False):
    dataset = Ts_Dataset(path, input_time_steps, output_time_steps, input_idx, output_idx, transform)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)


class Ts_Dataset(Dataset): #先变换数据，再选取列
    def __init__(self, path, input_time_steps, output_time_steps,  input_idx, output_idx, transform):
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp')
        df.drop('timestamp', axis=1, inplace=True)
        if transform:
            df = transform(df)
        if input_idx is None:
            self.input_idx = range(0, df.shape[1])
        else:
            self.input_idx = input_idx
        if output_idx is None:
            self.output_idx = range(0, df.shape[1])
        else:
            self.output_idx = output_idx
        self.its = input_time_steps
        self.ops = output_time_steps
        self.len = len(df) - self.its - self.ops + 1
        self.X = []
        self.y = []
        for i in range(self.len):
            X_sample = df.iloc[i:i + self.its, self.input_idx].to_numpy()
            y_sample = df.iloc[i + self.its:i + self.its + self.ops, self.output_idx].to_numpy()
            self.X.append(X_sample)
            self.y.append(y_sample)
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        df = None
        self.X = torch.tensor(data=self.X, dtype=torch.float32)
        self.y = torch.tensor(data=self.y, dtype=torch.float32)

    def __repr__(self):
        return f'The dataset for time_series\ninput_time_steps :{self.its}\noutput_time_steps :{self.ops}\ntotal items :{self.len}'

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

def Ts_Train(train_dataloader, model,
                            learning_rate, num_epochs,
                            validation_dataloader=None, pre_work=None):
    # 定义损失函数和优化器
    print('---------------------开始训练------------------------')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #执行模型权重预处理设定优化初始化点
    if pre_work:
        pre_work(model)
    # 存储训练和验证误差
    train_losses = []
    validation_losses = []
    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        avg_loss = 0.0; tot = 0
        for input_data, target_data in train_dataloader:
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            tot += 1
        print(f'epoch:{epoch}, train_loss:{round(avg_loss / tot,5)}', end=', ')
        train_losses.append(avg_loss/tot)

        # 验证模型
        model.eval()
        if validation_dataloader is not None:
            with torch.no_grad():
                avg_loss = 0.0; tot = 0
                for input_data, target_data in validation_dataloader:
                    output = model(input_data)
                    loss = criterion(output, target_data)
                    avg_loss += loss.item()
                    tot += 1
                validation_losses.append(avg_loss/tot)
                print(f'val_loss:{round(avg_loss / tot,5)}')
    print('---------------------训练结束------------------------')

    # 绘制训练和验证误差曲线
    fig, axes = plt.subplots(nrows=1,ncols=2)
    axes[0].plot(np.linspace(1,num_epochs,num_epochs), train_losses, label='Train Loss', color='blue')
    if validation_dataloader is not None:
        axes[0].plot(np.linspace(1,num_epochs,num_epochs), validation_losses, label='Validation Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].title.set_text('Training and Validation Loss')

    axes[1].plot(np.linspace(int(num_epochs*0.8)+1, num_epochs, num_epochs-int(num_epochs*0.8)),
                    train_losses[int(num_epochs*0.8): num_epochs], label='Train Loss', color='blue')
    if validation_dataloader is not None:
        axes[1].plot(np.linspace(int(num_epochs*0.8)+1, num_epochs, num_epochs-int(num_epochs*0.8)),
                        validation_losses[int(num_epochs*0.8): num_epochs], label='Validation Loss', color='orange')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].title.set_text('Training and Validation Loss in the End')
    plt.show()