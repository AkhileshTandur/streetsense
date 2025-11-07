import argparse, torch, torch.optim as optim, torch.nn.functional as F
from streetsense.data.loaders import make_loader
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer
from streetsense.utils import set_seed, AverageMeter, to_device, accuracy_from_logits

def run(args):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = make_loader(split='train', batch_size=64)
    val = make_loader(split='test', batch_size=64, shuffle=False)

    audio = AudioCNN().to(device)
    imu = IMUCNN().to(device)
    vision = VisionBackbone().to(device)
    # load selfsup if available
    import os
    if os.path.exists('out/selfsup.pt'):
        sd = torch.load('out/selfsup.pt', map_location=device)
        audio.load_state_dict(sd['audio'], strict=False)
        imu.load_state_dict(sd['imu'], strict=False)
        vision.load_state_dict(sd['vision'], strict=False)

    fusion = FusionTransformer(n_classes=2).to(device) # 2 classes in pseudo labels

    params = list(audio.parameters())+list(imu.parameters())+list(vision.parameters())+list(fusion.parameters())
    opt = optim.Adam(params, lr=1e-3)

    for epoch in range(args.epochs):
        audio.train(); imu.train(); vision.train(); fusion.train()
        loss_meter=AverageMeter(); acc_meter=AverageMeter()
        for batch in dl:
            batch = to_device(batch, device)
            za = audio(batch['audio']); zm = imu(batch['imu']); zv = vision(batch['frame'])
            logits = fusion(za, zm, zv)
            loss = F.cross_entropy(logits, batch['y'])
            opt.zero_grad(); loss.backward(); opt.step()
            acc = accuracy_from_logits(logits, batch['y'])
            loss_meter.update(loss.item(), n=batch['y'].size(0))
            acc_meter.update(acc, n=batch['y'].size(0))
        # val
        audio.eval(); imu.eval(); vision.eval(); fusion.eval()
        with torch.no_grad():
            vloss=AverageMeter(); vacc=AverageMeter()
            for batch in val:
                batch = to_device(batch, device)
                logits = fusion(audio(batch['audio']), imu(batch['imu']), vision(batch['frame']))
                loss = F.cross_entropy(logits, batch['y'])
                acc = accuracy_from_logits(logits, batch['y'])
                vloss.update(loss.item(), n=batch['y'].size(0)); vacc.update(acc, n=batch['y'].size(0))
        print(f"epoch {epoch+1}: train loss {loss_meter.avg:.4f} acc {acc_meter.avg:.3f} | val loss {vloss.avg:.4f} acc {vacc.avg:.3f}")
    torch.save({'audio': audio.state_dict(), 'imu': imu.state_dict(), 'vision': vision.state_dict(), 'fusion': fusion.state_dict()}, 'out/supervised.pt')

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=2)
    args=p.parse_args(); run(args)
