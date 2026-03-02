import torch
import numpy as np

def compute_clustering_metrics(model):
    # Extract scalars (d=1)
    e0 = model.embed.embed.weight[0].item()
    e1 = model.embed.embed.weight[1].item()
    w_q = model.attention.q.weight[0, 0].item()
    w_k = model.attention.k.weight[0, 0].item()

    # Scores for pairs
    s_00 = (w_q * e0) * (w_k * e0)
    s_11 = (w_q * e1) * (w_k * e1)
    s_01 = (w_q * e0) * (w_k * e1)
    s_10 = (w_q * e1) * (w_k * e0)

    s_same = (s_00 + s_11) / 2
    s_diff = (s_01 + s_10) / 2
    delta = s_same - s_diff

    return {'s_same': s_same, 's_diff': s_diff, 'delta': delta}
