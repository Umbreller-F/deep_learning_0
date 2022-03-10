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
    input_size=5
    hidden_size=5
    num_classes=1
    device=0
    learning_rate=0.01
    num_epochs=0
    train_loader=0

    # with open('dataset.data') as my_data:  # 读取文件部分
    #     lines = my_data.readlines()
    #     data = np.zeros((200, 5), dtype=float)
    #     label = np.zeros((200, 1), dtype=float)
    #     i = 0
    #     for line in lines:
    #         line = line.split(',')  # 以逗号分开
    #         # print(line)
    #         for j in range(5):
    #             data[i][j]=float(line[0])
    #         label[i]=float(line[1].replace("\n",''))
    #         i+=1
    #     # print(data,label)

    model = Net(input_size, hidden_size, num_classes)
    print(model)

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
