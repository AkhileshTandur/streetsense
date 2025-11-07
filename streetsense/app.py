import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from streetsense.data.loaders import make_loader
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer
from streetsense.utils import to_device

st.set_page_config(page_title="StreetSense", page_icon="🚗", layout="centered")
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

col1, col2 = st.columns(2)
with col1:
    batch_size = st.slider("Batch size", 8, 128, 32, step=8)
with col2:
    show_al = st.checkbox("Show Active Learning candidates", value=True)

if st.button("Run on test split"):
    dl = make_loader(split='test', batch_size=batch_size, shuffle=False)
    ps, ys, confs = [], [], []
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, 'cpu')
            logits = fusion(audio(batch['audio']), imu(batch['imu']), vision(batch['frame']))
            probs = torch.softmax(logits, dim=-1).numpy()
            ps.extend(probs.argmax(1).tolist())
            confs.extend(probs.max(1))
            ys.extend(batch['y'].numpy().tolist())

    acc = sum(int(p == y) for p, y in zip(ps, ys)) / len(ys)
    st.metric("Accuracy (pseudo-labels)", f"{acc:.3f}")

    # Table with first 30 predictions
    df = pd.DataFrame({"y_true": ys, "y_pred": ps, "confidence": confs})
    st.subheader("Sample Predictions")
    st.dataframe(df.head(30))

    # Metrics
    st.subheader("Classification Report")
    cr_text = classification_report(ys, ps, digits=3)
    st.text(cr_text)

    # Confusion Matrix
    cm = confusion_matrix(ys, ps)
    st.subheader("Confusion Matrix")
    st.write(cm)

    if show_al:
        try:
            al = pd.read_csv("out/active_candidates.csv").sort_values("entropy", ascending=False)
            st.subheader("Top Uncertain Samples (Active Learning)")
            st.dataframe(al.head(25))
        except Exception as e:
            st.info("Run the active learning step first to see candidates (python -m streetsense.active_learning --budget 50).")

st.caption("Models are trained on synthetic data for a fast demo.")
