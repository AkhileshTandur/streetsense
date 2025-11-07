import argparse, torch, torch.optim as optim
from streetsense.data.loaders import make_loader
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.selfsup.contrastive import nt_xent, augment_audio, augment_imu
from streetsense.utils import set_seed, AverageMeter, to_device

def run(args):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = make_loader(split='train', batch_size=64)
    audio_enc = AudioCNN().to(device)
    imu_enc = IMUCNN().to(device)
    vision_enc = VisionBackbone().to(device)
    opt = optim.Adam(list(audio_enc.parameters())+list(imu_enc.parameters())+list(vision_enc.parameters()), lr=1e-3)
    for epoch in range(args.epochs):
        meter = AverageMeter()
        for batch in dl:
            batch = to_device(batch, device)
            za1 = audio_enc(augment_audio(batch['audio']))
            za2 = audio_enc(augment_audio(batch['audio']))
            zm1 = imu_enc(augment_imu(batch['imu']))
            zm2 = imu_enc(augment_imu(batch['imu']))
            zv1 = vision_enc(batch['frame'])
            zv2 = vision_enc(batch['frame'])
            loss = nt_xent(za1, za2) + nt_xent(zm1, zm2) + nt_xent(zv1, zv2)
            opt.zero_grad(); loss.backward(); opt.step()
            meter.update(loss.item(), n=1)
        print(f"epoch {epoch+1} contrastive loss: {meter.avg:.4f}")
    torch.save({'audio': audio_enc.state_dict(),
                'imu': imu_enc.state_dict(),
                'vision': vision_enc.state_dict()}, 'out/selfsup.pt')

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=2)
    args=p.parse_args(); run(args)
