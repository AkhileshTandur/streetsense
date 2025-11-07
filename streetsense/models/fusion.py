import torch, torch.nn as nn, torch.nn.functional as F

class FusionTransformer(nn.Module):
    def __init__(self, d=128, nhead=4, nlayers=2, n_classes=4):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Linear(d, n_classes)
    def forward(self, a, m, v): # a,m,v: [B,d]
        x = torch.stack([a,m,v], dim=1) # [B,3,d]
        h = self.enc(x)[:,0]            # use first token
        logits = self.cls(h)
        return logits
