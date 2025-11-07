import torch, torch.nn as nn, torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,16,5,2,2), nn.ReLU(), nn.Conv1d(16,32,5,2,2), nn.ReLU(),
            nn.Conv1d(32,64,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(64, out_dim)
    def forward(self, x): # x: [B,1,T]
        h = self.net(x).squeeze(-1)
        z = self.proj(h)
        return F.normalize(z, dim=-1)

class IMUCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6,32,5,2,2), nn.ReLU(), nn.Conv1d(32,64,5,2,2), nn.ReLU(),
            nn.Conv1d(64,128,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, out_dim)
    def forward(self, x): # [B,6,T]
        h = self.net(x).squeeze(-1)
        z = self.proj(h)
        return F.normalize(z, dim=-1)

class VisionBackbone(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(64, out_dim)
    def forward(self, x): # [B,3,H,W]
        h = self.net(x).flatten(1)
        z = self.proj(h)
        return F.normalize(z, dim=-1)
