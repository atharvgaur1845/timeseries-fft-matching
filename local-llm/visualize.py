import torch
from model_llm_v6 import MultiHead_Model 
model = MultiHead_Model(input_len=1024, d_model=512, feature_dim=47)
dummy_x = torch.randn(4, 1024)
dummy_feats = torch.randn(4, 47)
torch.onnx.export(
    model,                          
    (dummy_x, dummy_feats),   
    "multihead_model.onnx",
    input_names=['signal', 'features'],
    output_names=['output', 'pred_stats'],
    opset_version=14
)
