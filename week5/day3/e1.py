import torch

layers = 32
heads = 32
head_dim = 128

for seq in [512, 2048, 8192]:
    size = layers * heads * seq * head_dim * 2  # K & V
    print(seq, "tokens ->", size/1e6, "MB")