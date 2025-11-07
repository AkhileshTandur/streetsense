import torch
from streetsense.models.encoders import AudioCNN, IMUCNN, VisionBackbone
from streetsense.models.fusion import FusionTransformer

def test_encoder_shapes():
    a = AudioCNN(); m = IMUCNN(); v = VisionBackbone()
    za = a(torch.randn(2,1,512)); zm = m(torch.randn(2,6,128)); zv = v(torch.randn(2,3,64,64))
    assert za.shape == (2,128)
    assert zm.shape == (2,128)
    assert zv.shape == (2,128)
    f = FusionTransformer(n_classes=2)
    logits = f(za, zm, zv)
    assert logits.shape == (2,2)
