import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    e = torch.abs(y_pred-y_true)
    return torch.mean(e)

def mape(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    e = torch.abs(y_pred-y_true)
    return torch.mean(e/torch.abs(y_true))


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
                            validation_dataloader=None, pre_work=None, save_results=False, log_process=False):
    # 定义损失函数和优化器
    print('---------------------开始训练------------------------')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.MSELoss().to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #执行模型权重预处理设定优化初始化点
    if pre_work:
        model = pre_work(model)
    # 存储训练和验证误差
    train_losses = []; validation_losses = []
    validation_mae = []
    validation_mape= []
    for epoch in range(num_epochs):
        # 训练模型
        model.train()
        avg_loss = torch.tensor(data=0.0, device=device)
        tot = torch.tensor(data=0, device=device)
        for input_data, target_data in train_dataloader:
            optimizer.zero_grad()
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            output = model(input_data)
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            tot += 1
        Mse = (avg_loss / tot).cpu().item()
        print(f'epoch:{epoch}, train_loss:{round(Mse,5)}', end=', ')
        train_losses.append(Mse)
        # 验证模型
        if validation_dataloader is not None:
            model.eval()
            with torch.no_grad():
                avg_loss = torch.tensor(data=0.0, device=device)
                tot = torch.tensor(data=0, device=device)
                avg_mae = torch.tensor(data=0.0, device=device)
                avg_mape = torch.tensor(data=0.0, device=device)
                for input_data, target_data in validation_dataloader:
                    input_data = input_data.to(device)
                    target_data = target_data.to(device)
                    output = model(input_data)
                    loss = criterion(output, target_data)
                    avg_loss += loss.item()
                    avg_mae += mae(output, target_data)
                    avg_mape += mape(output, target_data)
                    tot += 1
                Mse = (avg_loss / tot).cpu().item()
                Mae = (avg_mae / tot).cpu().item()
                Mape = (avg_mape / tot).cpu().item()
                validation_losses.append(Mse)
                validation_mae.append(Mae)
                validation_mape.append(Mape)
                print(f'val_loss:{round(Mse, 5)}, val_mae:{round(Mae,5)}, val_mape:{round(Mape,5)}')
    print('---------------------训练结束------------------------')
    #抛弃第一个epoch的输出，因为还没有进行任何学习，输出的结果没有意义
    train_losses = train_losses[1:]; validation_losses = validation_losses[1:]
    validation_mape= validation_mape[1:]; validation_mae = validation_mae[1:]
    num_epochs -= 1
    # 绘制训练和验证误差曲线
    fig, axes = plt.subplots(nrows=1,ncols=2)
    axes[0].plot(np.linspace(1,num_epochs,num_epochs), train_losses, label='MSE (Train)', color='darkblue')
    if validation_dataloader is not None:
        axes[0].plot(np.linspace(1,num_epochs,num_epochs), validation_losses, label='MSE (Validation)', color='royalblue')
        axes[0].plot(np.linspace(1, num_epochs, num_epochs), validation_mae, label='MAE (Validation)', color='lightcoral')
        axes[0].plot(np.linspace(1, num_epochs, num_epochs), validation_mape, label='MAPE (Validation)', color='goldenrod')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].title.set_text('Training and Validation')

    axes[1].plot(np.linspace(int(num_epochs*0.8)+1, num_epochs, num_epochs-int(num_epochs*0.8)),
                    train_losses[int(num_epochs*0.8): num_epochs], label='MSE (Train)', color='darkblue')
    if validation_dataloader is not None:
        axes[1].plot(np.linspace(int(num_epochs*0.8)+1, num_epochs, num_epochs-int(num_epochs*0.8)),
                        validation_losses[int(num_epochs*0.8): num_epochs], label='MSE (Validation)', color='royalblue')
        axes[1].plot(np.linspace(int(num_epochs * 0.8) + 1, num_epochs, num_epochs - int(num_epochs * 0.8)),
                     validation_mae[int(num_epochs * 0.8): num_epochs], label='MAE (Validation)', color='lightcoral')
        axes[1].plot(np.linspace(int(num_epochs * 0.8) + 1, num_epochs, num_epochs - int(num_epochs * 0.8)),
                     validation_mape[int(num_epochs * 0.8): num_epochs], label='MAPE (Validation)', color='goldenrod')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].title.set_text('Training and Validation in the End')
    plt.show()
    if save_results:
        torch.save(model.state_dict(), f'{model.__class__.__name__ }_{num_epochs+1}epochs.pth')
        print('---------------------模型权重已保存------------------------')
    if log_process:
        data = np.array([train_losses,validation_losses,validation_mae,validation_mape]).T
        data = pd.DataFrame(data=data, columns=['train_losses', 'validation_losses', 'validation_mae', 'validation_mape'])
        data.to_csv(f'{model.__class__.__name__ }_{num_epochs+1}epochs.csv', index=False)
        print('---------------------训练记录已保存------------------------')
