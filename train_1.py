import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
# plt.switch_backend('agg')


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, n_output):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.activate = nn.Tanh()
        # self.activate = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,hidden_size)
        self.fc5 = nn.Linear(hidden_size,hidden_size)
        # self.af2 = nn.Tanh()
        self.fc0 = nn.Linear(hidden_size,n_output)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.activate(out)
        out = self.fc2(out)
        out = self.activate(out)
        out = self.fc3(out)
        out = self.activate(out)
        out = self.fc4(out)
        out = self.activate(out)
        out = self.fc5(out)
        out = self.activate(out)
        out = self.fc0(out)
        return out

class Dataset(Data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        img, target = self.inputs[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.inputs)


if __name__=="__main__":
    # 网络参数
    input_size=1
    hidden_size=10
    n_output=1
    batch_size=5
    training_size=20000

    model = Net(input_size, hidden_size, n_output)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.L1Loss()

    with open('dataset.data') as my_data:  # 读取文件部分
        lines = my_data.readlines()
        data = np.zeros((training_size,1), dtype=float)
        label = np.zeros((training_size,1), dtype=float)
        # print(data)
        i = 0
        for i in range(training_size):
            line = lines[i].split(',')  # 以逗号分开
            # print(line)
            data[i][0]=float(line[0])
            label[i][0]=float(line[1].replace("\n",''))
            i+=1
        # print(data,label)

    data = torch.Tensor(data)
    label = torch.Tensor(label)
    # print(data)
    # print(model(data))

    training_set = Data.TensorDataset(data, label)

    loader = Data.DataLoader(
        dataset=training_set,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        # num_workers=2,  # 多线程来读数据
    )

    for epoch in range(10):
        for step, (batch_x, batch_y) in enumerate(loader):
            # print(batch_x)
            prediction = model(batch_x)
            # print(prediction)
            loss = loss_func(prediction,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        x=torch.unsqueeze(torch.linspace(0, 4, 101),dim=1)
        x=x*np.pi
        plt.style.use('ggplot')
        plt.plot(x.numpy(), model(x).detach().numpy())
        plt.title("epoch:{}".format(epoch))
        plt.show()
    
    x=torch.unsqueeze(torch.linspace(0, 4, 101),dim=1)
    x=x*np.pi
    plt.style.use('ggplot')
    plt.plot(x.numpy(), model(x).detach().numpy())
    plt.title("result")
    plt.show()

    torch.save(model, 'net1.pkl')
    torch.save(model.state_dict(),'net1_params.pkl')









    # for _ in range(200):
    #     prediction = model(x)
    #     loss = loss_func(prediction,y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    # x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
    # y = x.pow(2) + 0.2*torch.rand(x.size())

    # x, y = Variable(x), Variable(y)

    # plt.scatter(x.data.numpy(), y.data.numpy())
    # plt.show()

    # x = torch.linspace(-1,1,100)
    # print(x)