import torch

width = 14
height = 14
x = torch.arange(width, dtype=torch.long)
y = torch.arange(height, dtype=torch.long)
z = torch.stack(torch.meshgrid([x, y]), dim=-1).view(width * height, 2)

print(z)

