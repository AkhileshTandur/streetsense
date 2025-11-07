import torch
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer

# Load trained weights
device = 'cpu'
ckpt = torch.load('out/supervised.pt', map_location=device)

audio = AudioCNN()
imu = IMUCNN()
vision = VisionBackbone()
fusion = FusionTransformer(n_classes=2)

audio.load_state_dict(ckpt['audio'])
imu.load_state_dict(ckpt['imu'])
vision.load_state_dict(ckpt['vision'])
fusion.load_state_dict(ckpt['fusion'])

audio.eval()
imu.eval()
vision.eval()
fusion.eval()

# Example dummy inputs (shapes must match training)
xa = torch.randn(1, 1, 512)
xm = torch.randn(1, 6, 128)
xv = torch.randn(1, 3, 64, 64)

# Convert encoders to TorchScript for edge/ONNX export
sa = torch.jit.trace(audio, xa)
sm = torch.jit.trace(imu, xm)
sv = torch.jit.trace(vision, xv)

# Save to 'out/' directory
sa_path = "out/audio_encoder.ts"
sm_path = "out/imu_encoder.ts"
sv_path = "out/vision_encoder.ts"

torch.jit.save(sa, sa_path)
torch.jit.save(sm, sm_path)
torch.jit.save(sv, sv_path)

print("✅ Saved TorchScript models to:")
print("  ", sa_path)
print("  ", sm_path)
print("  ", sv_path)
