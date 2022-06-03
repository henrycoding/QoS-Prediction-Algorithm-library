import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, type_, embedding_nums: list, embedding_dims: list):
        self.type = type_
        self.embedding_nums = embedding_nums
        self.embedding_dims = embedding_dims
        assert self.type in ["stack", "cat"]
        super().__init__()
        self.embeddings = nn.ModuleList([
            *[
                nn.Embedding(num, dim)
                for num, dim in zip(embedding_nums, embedding_dims)
            ]
        ])

    def forward(self, indexes):
        if self.type == "stack":
            assert len(set(
                self.embedding_dims)) == 1, f"dims should be the same"

            x = sum([
                embedding(indexes[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ])
        elif self.type == "cat":
            x = torch.cat([
                embedding(indexes[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ],
                          dim=1)
        else:
            raise NotImplementedError
        return x