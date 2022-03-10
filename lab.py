import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
plt.switch_backend('agg')

print(torch.linspace(-1, 1, 100))
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
print(x.numpy())