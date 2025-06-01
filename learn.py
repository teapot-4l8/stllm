# import torch.nn as nn
# import torch

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Linear(10, 2)  # This layer takes input of size 10 and outputs size 2.
#     def forward(self, x):
#         return self.fc(x)

# model = SimpleNet()
# x = torch.randn(8, 10)   # 8 samples, 10 features each
# output = model(x)        # output shape: [8, 2]



# import torch

# # Create a random tensor with shape [2, 3, 4, 3]
# x = torch.arange(2*3*4*3).reshape(2, 3, 4, 3)

# # Print x for clarity
# print("x shape:", x.shape)
# print(x)

# day_emb = x[..., 1]
# print("day_emb shape:", day_emb.shape)  # day_emb shape: torch.Size([2, 3, 4])
# print(day_emb)


import torch

B, S, F, T = 2, 3, 2, 4
input_data = torch.arange(B * S * F * T).reshape(B, S, F, T)
print("Original shape:", input_data.shape)
print(input_data)

# Reshape and transform
input_data = input_data.view(B, S, -1) 
print("\nAfter view:", input_data.shape)  # torch.Size([2, 3, 8])
print(input_data)

input_data = input_data.transpose(1, 2)  # 转置
print("\nAfter transpose:", input_data.shape)
print(input_data)

input_data = input_data.unsqueeze(-1)
print("\nAfter unsqueeze:", input_data.shape)  #  torch.Size([2, 8, 3, 1])
print(input_data)