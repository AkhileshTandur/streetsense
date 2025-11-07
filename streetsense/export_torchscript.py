import torch
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer

device='cpu'
ckpt=torch.load('out/supervised.pt', map_location=device)
a, m, v = AudioCNN(), IMUCNN(), VisionBackbone()
f = FusionTransformer(n_classes=2)
a.load_state_dict(ckpt['audio']); m.load_state_dict(ckpt['imu']); v.load_state_dict(ckpt['vision']); f.load_state_dict(ckpt['fusion'])
a.eval(); m.eval(); v.eval(); f.eval()

# example inputs
xa = torch.randn(1,1,512)
xm = torch.randn(1,6,128)
xv = torch.randn(1,3,64,64)

sa = torch.jit.trace(a, xa); sm = torch.jit.trace(m, xm); sv = torch.jit.trace(v, xv)
torch.jit.save(sa, "out/audio_encoder.ts")
torch.jit.save(sm, "out/imu_encoder.ts")
torch.jit.save(sv, "out/vision_encoder.ts")
print("Saved TorchScript files to out/")
