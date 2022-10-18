import torch

import time



dev = 'mps:0'

conv = torch.nn.Conv2d(10, 10, 3).to(dev)

img = torch.randn(64, 10, 64, 64).to(dev)



t0 = time.time()

for i in range(1000):

    conv(img)

t1 = time.time()

print('Use mps, time:{}'.format(t1-t0))



dev = 'cpu'

conv = torch.nn.Conv2d(10, 10, 3).to(dev)

img = torch.randn(64, 10, 64, 64).to(dev)



t0 = time.time()

for i in range(1000):

    conv(img)

t1 = time.time()

print('Use cpu, time:{}'.format(t1-t0))
