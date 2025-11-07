import streamlit as st
import torch
import torch.nn.functional as F
from streetsense.data.loaders import make_loader
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer
from streetsense.utils import to_device

st.title("🚗 StreetSense — Multimodal Road-Defect Demo")

@st.cache_resource
def load_models():
    device = 'cpu'
    ckpt = torch.load('out/supervised.pt', map_location=device)
    audio, imu, vision = AudioCNN(), IMUCNN(), VisionBackbone()
    fusion = FusionTransformer(n_classes=2)
    audio.load_state_dict(ckpt['audio'])
    imu.load_state_dict(ckpt['imu'])
    vision.load_state_dict(ckpt['vision'])
    fusion.load_state_dict(ckpt['fusion'])
    for m in (audio, imu, vision, fusion):
        m.eval()
    return audio, imu, vision, fusion

audio, imu, vision, fusion = load_models()

batch_size = st.slider("Batch size", 8, 128, 32)

if st.button("Run on test split"):
    dl = make_loader(split='test', batch_size=batch_size, shuffle=False)
    ps, ys = [], []
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, 'cpu')
            logits = fusion(audio(batch['audio']), imu(batch['imu']), vision(batch['frame']))
            ps.extend(logits.argmax(-1).numpy().tolist())
            ys.extend(batch['y'].numpy().tolist())
    acc = sum(int(p == y) for p, y in zip(ps, ys)) / len(ys)
    st.metric("Accuracy (pseudo-labels)", f"{acc:.3f}")
