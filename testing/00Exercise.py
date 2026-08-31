"""Original matrix multiplication exercise."""

import torch

torch.manual_seed(0)
random_tensor = torch.rand(7, 7)
random_tensor_2 = torch.rand(1, 7)
answer_tensor = torch.mm(random_tensor, random_tensor_2.T)
print(answer_tensor)

