import torch
import torch.nn as nn
from torchviz import make_dot
from model_llm_v6 import MultiHead_Model, StatisticalMatchingHead
BATCH_SIZE = 4
WINDOW_SIZE = 1024
FEATURE_COUNT = 47
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 8
D_FF = 1024
DROPOUT = 0.4
model = MultiHead_Model(
    input_len=WINDOW_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
    feature_dim=FEATURE_COUNT
)
dummy_x = torch.randn(BATCH_SIZE, WINDOW_SIZE)
dummy_feats = torch.randn(BATCH_SIZE, FEATURE_COUNT)
output, pred_stats = model(dummy_x, dummy_feats)
graph = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
graph.render('model_llm_v6_graph', format='png', cleanup=True)