import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
# import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__=="__main__":
    # 参数设置
    input_size=0
    hidden_size=0
    num_classes=0
    device=0
    learning_rate=0
    num_epochs=0

    with open('dataset.data') as my_data:  # 读取文件部分
        lines = my_data.readlines()
        data = np.zeros((200, 5), dtype=float)
        label = np.zeros((200, 1), dtype=float)
        i = 0
        for line in lines:
            line = line.split(',')  # 以逗号分开
            # print(line)
            for j in range(5):
                data[i][j]=float(line[0])
            label[i]=float(line[1].replace("\n",''))
            i+=1
        # print(data,label)

    # 数据集分割比
    ratioTraining = 0.4
    ratioValidation = 0.1
    ratioTesting = 0.5
    # x为数据，y为标签
    xTraining, xTesting, yTraining, yTesting = train_test_split(data, label, test_size=1 - ratioTraining,
                                                                random_state=0)  # 随机分配数据集
    xTesting, xValidation, yTesting, yValidation = train_test_split(xTesting, yTesting,
                                                                    test_size=ratioValidation / ratioTesting,
                                                                    random_state=0)  # Q2:比例好像不太对？
    print(xTesting.shape)
    print(xValidation.shape)
    # 拆分成测试集和验证集

    # scaler = StandardScaler(copy=False)
    # scaler.fit(xTraining)  # 求均值和方差
    # scaler.transform(xTraining)  # 标准归一化
    # scaler.transform(xTesting)  # Q1:按训练集的均值和方差做归一化？
    # scaler.transform(xValidation)

    # model = Net(input_size, hidden_size, num_classes).to(device)

    # # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # # Train the model
    # total_step = len(train_loader)
    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(train_loader):  
    #         # Move tensors to the configured device
    #         images = images.reshape(-1, 28*28).to(device)
    #         labels = labels.to(device)
    #         # Forward pass
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    print(1)