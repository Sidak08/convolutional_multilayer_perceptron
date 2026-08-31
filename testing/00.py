"""Original tensor basics exercise."""

import torch

scalar = torch.tensor(7)
vector = torch.tensor([1, 3])
matrix = torch.tensor([[1, 2], [1, 2]])
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

print(scalar, vector, matrix.ndim, tensor.shape)
random_tensor = torch.rand(2, 3)
zeros = torch.zeros_like(random_tensor)
print(random_tensor + zeros)
print(torch.matmul(torch.randint(0, 10, (2, 2)), torch.randint(0, 10, (2, 2))))
torch.manual_seed(1234)

