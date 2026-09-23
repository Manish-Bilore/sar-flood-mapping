"""gpu_check.py — measure what trains on the P520 (2 GB, sm_61, FP32 only).
Pascal GP108 has ~1/64-rate FP16 -> AMP gives no speedup; memory saving only."""
import time, torch, segmentation_models_pytorch as smp

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print(f"device={dev}", torch.cuda.get_device_name(0) if dev == "cuda" else "")

CANDIDATES = [  # (name, ctor)
    ("Unet-resnet18",     lambda c: smp.Unet("resnet18", encoder_weights=None, in_channels=c, classes=1)),
    ("Unet-resnet34",     lambda c: smp.Unet("resnet34", encoder_weights=None, in_channels=c, classes=1)),
    ("Unet++-resnet34",   lambda c: smp.UnetPlusPlus("resnet34", encoder_weights=None, in_channels=c, classes=1)),
    ("DeepLabV3+-r34",    lambda c: smp.DeepLabV3Plus("resnet34", encoder_weights=None, in_channels=c, classes=1)),
    ("Segformer-mit_b0",  lambda c: smp.Segformer("mit_b0", encoder_weights=None, in_channels=c, classes=1)),
    ("Segformer-mit_b2",  lambda c: smp.Segformer("mit_b2", encoder_weights=None, in_channels=c, classes=1)),
]

def bench(ctor, chans, size, bs, amp, steps=10):
    m = ctor(chans).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.randn(bs, chans, size, size, device=dev); y = (torch.rand(bs, 1, size, size, device=dev) > .8).float()
    lossf = torch.nn.BCEWithLogitsLoss(); scaler = torch.amp.GradScaler(enabled=amp)
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 2):
        if i == 2:
            if dev == "cuda": torch.cuda.synchronize()
            t = time.time()
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            loss = lossf(m(x), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    if dev == "cuda": torch.cuda.synchronize()
    dt = (time.time() - t) / steps
    mem = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    del m, opt; torch.cuda.empty_cache() if dev == "cuda" else None
    return dt, mem

# 8 ch = UrbanSARFloods-style stack (pre/co intensity VV,VH + coherence VV,VH ...); 2 ch = post VV,VH
print(f"{'model':18s} {'ch':>2s} {'px':>4s} {'bs':>3s} {'amp':>4s} {'s/iter':>7s} {'peakGB':>7s} {'img/s':>6s}")
for name, ctor in CANDIDATES:
    for chans, size, bs in [(2, 256, 8), (8, 256, 8), (8, 512, 2)]:
        for amp in (False, True):
            try:
                dt, mem = bench(ctor, chans, size, bs, amp)
                print(f"{name:18s} {chans:2d} {size:4d} {bs:3d} {str(amp):>4s} {dt:7.3f} {mem:7.2f} {bs/dt:6.1f}")
            except torch.OutOfMemoryError:
                print(f"{name:18s} {chans:2d} {size:4d} {bs:3d} {str(amp):>4s}     OOM"); torch.cuda.empty_cache()
            except Exception as e:
                print(f"{name:18s} {chans:2d} {size:4d} {bs:3d} {str(amp):>4s}   ERR {type(e).__name__}: {str(e)[:60]}")
