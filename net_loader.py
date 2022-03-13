import matplotlib.pyplot as plt
import torch
import numpy as np
from train_1 import Net


x=torch.unsqueeze(torch.linspace(0, 4, 101),dim=1)
x=x*np.pi

model_copy1=torch.load('net1.pkl')
plt.style.use('ggplot')
plt.plot(x.numpy(), model_copy1(x).data.numpy())
plt.title("copy1")
plt.show()

model_copy2=Net(1, 10, 1)
model_copy2.load_state_dict(torch.load('net1_params.pkl'))
plt.style.use('ggplot')
plt.plot(x.numpy(), model_copy1(x).detach().numpy())
plt.title("copy2")
plt.show()
