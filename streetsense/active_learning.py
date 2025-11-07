import argparse, torch, torch.nn.functional as F
from streetsense.data.loaders import make_loader
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer
from streetsense.utils import to_device
import numpy as np, pandas as pd, os

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = make_loader(split='train', batch_size=128)
    ckpt = torch.load('out/supervised.pt', map_location=device)
    audio=AudioCNN().to(device); audio.load_state_dict(ckpt['audio'])
    imu=IMUCNN().to(device); imu.load_state_dict(ckpt['imu'])
    vision=VisionBackbone().to(device); vision.load_state_dict(ckpt['vision'])
    fusion=FusionTransformer(n_classes=2).to(device); fusion.load_state_dict(ckpt['fusion'])
    audio.eval(); imu.eval(); vision.eval(); fusion.eval()

    candidates = []
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, device)
            logits = fusion(audio(batch['audio']), imu(batch['imu']), vision(batch['frame']))
            probs = F.softmax(logits, dim=-1)
            ent = -(probs*probs.log()).sum(-1).cpu().numpy()
            for e in ent: candidates.append(float(e))
    idx = np.argsort(candidates)[-args.budget:]
    pd.DataFrame({'sample_index': idx, 'entropy': np.array(candidates)[idx]}).to_csv('out/active_candidates.csv', index=False)
    print(f"Selected {len(idx)} samples for labeling (highest entropy). Saved to out/active_candidates.csv")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--budget', type=int, default=50)
    args=p.parse_args(); run(args)
