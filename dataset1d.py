# %%
import matplotlib.pyplot as plt
import numpy as np
import torch


class Dataset1D:
    def __init__(self):
        self.a, self.b = torch.load("/home/tianhao/1ddata.pt")
        self.full_data = torch.concatenate((self.a, self.b)).reshape(2, -1)
    
    def __len__(self):
        return 8192

    def __getitem__(self, idx):
        return self.full_data

data = Dataset1D()
gt = data[0]

# length = 32
# a = ((torch.arange(length) - length/2.0)/(length/2))**2 + 0.5* torch.rand(length)
# b = ((torch.arange(length) - length/2.0)/(length/2))**3 + 0.5* torch.rand(length)

# plt.plot(a, b, 'o')
# plt.show()

# torch.save((a,b), "1ddata.pt")
# a, b = torch.load("1ddata.pt")
# plt.plot(a, b, 'o')
# plt.show()
# #%%
# dataset = Dataset1D()
# plt.plot(dataset[0][0], dataset[0][1], "o")
# # %%
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# # %%
# for batch in dataloader:
#     print(batch.shape)
#     break
# # %%