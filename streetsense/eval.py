import torch, argparse
from sklearn.metrics import classification_report, confusion_matrix
from streetsense.data.loaders import make_loader
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer
from streetsense.utils import to_device

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = make_loader(split='test', batch_size=128, shuffle=False)
    ckpt = torch.load('out/supervised.pt', map_location=device)

    audio, imu, vision = AudioCNN().to(device), IMUCNN().to(device), VisionBackbone().to(device)
    fusion = FusionTransformer(n_classes=2).to(device)
    audio.load_state_dict(ckpt['audio']); imu.load_state_dict(ckpt['imu'])
    vision.load_state_dict(ckpt['vision']); fusion.load_state_dict(ckpt['fusion'])
    audio.eval(); imu.eval(); vision.eval(); fusion.eval()

    ys, ps = [], []
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, device)
            logits = fusion(audio(batch['audio']), imu(batch['imu']), vision(batch['frame']))
            ps.extend(logits.argmax(-1).cpu().numpy().tolist())
            ys.extend(batch['y'].cpu().numpy().tolist())

    print(classification_report(ys, ps, digits=3))
    print("Confusion matrix:\n", confusion_matrix(ys, ps))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); _ = parser.parse_args()
    main()
