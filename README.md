# HuiChu_TimeSeries-HCTS
A Time Series Library Written by Python for commemorating Huichu.

In order to train and test my own time series model more conveniently, and at the same time to exercise my coding skills, this library only encapsulates the parts of dataset generation and model training and testing.

**You can put your model into a training and testing function called Ts_Train only if the model is based on torch.nn.Module.**

Use **Ts_Train** to 
* **Train the model**
* **Load pre-trained weights**
* **Check training process of the model**
* **Visualize the performance of model in training and testing**
* **Save the trained model as *.pth* files**
* **Log the training and testing process as *.csv* files**
  
You only need to download the **hcts.py** and use Ts_Train; maybe you can create more functions based on my code, just for your specific requirements.

**The following tutorial will teach you how to use HCTS.**

### First 
you have to change the name of your dataset's first column to ***timestamp***. Other columns have no requirements.
|timestamp |  close  |	open	 |  high	 |  low    |
| -------- | --------| ------- | --------| --------|
|2022/12/30|	6.8972 |	6.9595 |	6.9618 | 	6.8931 |
|2022/12/29|	6.9625 |	6.9805 |	6.9814 |	6.9611 |
|2022/12/28|	6.9774 |	6.963  |	6.9787 |	6.9616 |
|2022/12/27|	6.96	 |  6.9629 |	6.9684 |	6.949  |

### Second
Use ***Ts_Dateloader*** to make train_dataloader and test_dataloader for training and testing your model respectively.

The arguments of Ts_Dataloader are as follows:
* **input_time_steps**: *Length of sequence input to the model*
* **output_time_steps**: *Length of sequence the model output*
* **input_idx**: *Select columns you want to input the model by indicating the index of them in dataset. e.g. the index of the close column in the above dataset is 0. the open is 1. the high is 2. the low is 3. Defalut indicate you wanto input all columns*
* **output_idx**: *Select columns the model output by indicating the index of them in dataset. e.g. the index of the close column in the above dataset is 0. the open is 1. the high is 2. the low is 3. Defalut indicate the model output all columns*
* **shuffle**: *Whether you wanto shuffle the dataset*
* **transform**: *A function that transform the dataset before main process. e.g. if you want to standardize the dataset, you can write a function as below and transmit the function to *Ts_Dataloader*.*
```python
def tras(df: pandas.DataFrame) -> pandas.DataFrame:
    return (df - df.mean())/(df.std()+1e-6)
```
***The input and output type of the transform function are only Pandas.DataFrame!***

```python
td = Ts_Dataloader('data/train_data.csv', input_time_steps=7, output_time_steps=1, output_idx=0, batch_size=128,
                   shuffle=True, transform=tras)# train_dataloader
vd = Ts_Dataloader('data/test_data.csv',input_time_steps=7,output_time_steps=1,output_idx=0,batch_size=128,
                   shuffle=True, transform=tras)# valid_dataloader
```
You can use ***for x,y in td*** to check whether the data is generated as you want it to be
```python
for x,y in td:
  print(x, y)
```
### Third
**After dataloader, you have to prepare your model.** Here, take LSTM as an example for a demonstration.
```python
class lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=128, num_layers=3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 1)

    def forward(self, X):
        out, _ = self.lstm(X)
        out = self.fc(out[:, -1, :])
        return out
```
### Fourth
Use **Ts_Train** to 
* **Train the model**
* **Load pre-trained weights**
* **Check training process of the model**
* **Visualize the performance of model in training and testing**
* **Save the trained model as *pth* files**
* **Log the training and testing process as *csv* files**

```python
model = lstm()
Ts_Train(model=model, learning_rate=1e-3, num_epochs=50, train_dataloader=td,
         validation_dataloader=vd, save_results=True, log_process=True, pre_work=prework)
```
The arguments of Ts_Train are as follows:
* **log_process**: *whether log the training and testing process as *.csv* files*
* **save_results**: *whether save the trained model as *.pth* files*
* **pre_work**: *A function to manipulate model before training process. e.g. You can use the **prework** function to load pre_trained weights*
  
```python
def prework(model: nn.Module) -> nn.Module:
    state_dict = torch.load('lstm_50epochs.pth')
    model.load_state_dict(state_dict)
    return model
```

***The input and output type of the prework function are only torch.nn.Module!***

### Fifth
**Full code**
```python
import pandas
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

def tras(df: pandas.DataFrame) -> pandas.DataFrame:
    return (df - df.mean())/(df.std()+1e-6)

def prework(model: nn.Module) -> nn.Module:
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

```
