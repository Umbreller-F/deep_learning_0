from ast import If
from unittest import main
import torch
import math
import random

with open('dataset.data','w') as f:
    for _ in range(200):
        x=random.uniform(0, 4*math.pi)
        f.write(str(x))
        f.write(',')
        y=math.sin(x)+math.cos(x)
        f.write(str(y))
        f.write('\n')
