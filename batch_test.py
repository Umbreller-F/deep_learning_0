import torch
import torch.utils.data as Data
import numpy as np


torch.manual_seed(1)

BATCH_SIZE = 2  # 批训练的数据个数

x = torch.linspace(0, 4, 11)  # x data (torch tensor)
y = torch.linspace(10, 1, 10)  # y data (torch tensor)
print(x.numpy())
x = x*np.pi
print(x.numpy())

# # 先转换成 torch 能识别的 Dataset
# torch_dataset = Data.TensorDataset(x, y)

# # 把 dataset 放入 DataLoader
# loader = Data.DataLoader(
#     dataset=torch_dataset,  # torch TensorDataset format
#     batch_size=BATCH_SIZE,  # mini batch size
#     shuffle=True,  # 要不要打乱数据 (打乱比较好)
#     # num_workers=2,  # 多线程来读数据
# )

# for epoch in range(3):  # 训练所有!整套!数据 3 次
#     for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
#         # 假设这里就是你训练的地方...

#         # 打出来一些数据
#         # print('批次Epoch: ', epoch, '| Step: ', step, '| 数据batch x: ',
#         #       batch_x.numpy(), '| y: ', batch_y.numpy())
#         print(step, batch_x)