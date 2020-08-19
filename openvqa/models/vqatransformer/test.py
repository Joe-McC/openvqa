import torch

width = 3
height = 3
x = torch.arange(width, dtype=torch.long)
y = torch.arange(height, dtype=torch.long)
z = torch.stack(torch.meshgrid([x, y]), dim=-1).view(width * height, 2)

a = torch.ones((3,9,2))
print(z.shape)
print(z)
print(a+z)
