"""Local standard FoundationStereo loader using the established Windows SDPA shim."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import types


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model():
    os.environ["XFORMERS_DISABLED"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.dont_write_bytecode = True
    import torch
    import torch.nn.functional as F
    import timm
    from omegaconf import OmegaConf

    def attention(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                  window_size=(-1, -1), **kwargs):
        if kwargs or causal:
            raise ValueError("Unsupported attention options")
        mask = None
        if window_size != (-1, -1):
            nq, nk = q.shape[1], k.shape[1]
            qi = torch.arange(nq, device=q.device)[:, None] + nk - nq
            ki = torch.arange(nk, device=q.device)[None, :]
            mask = torch.ones((nq, nk), dtype=torch.bool, device=q.device)
            if window_size[0] >= 0:
                mask &= ki >= qi - window_size[0]
            if window_size[1] >= 0:
                mask &= ki <= qi + window_size[1]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask, dropout_p=dropout_p, scale=softmax_scale).transpose(1, 2)

    shim = types.ModuleType("flash_attn")
    shim.flash_attn_func = attention
    shim.flash_attn_qkvpacked_func = lambda qkv, **kw: attention(*qkv.unbind(2), **kw)
    sys.modules["flash_attn"] = shim
    original_create = timm.create_model

    def local_create(*a, **kw):
        kw["pretrained"] = False
        return original_create(*a, **kw)

    def local_hub(repo, model, *a, **kw):
        if repo != "facebookresearch/dinov2":
            raise ValueError(f"Unexpected hub source: {repo}")
        from dinov2.hub import backbones
        kw["pretrained"] = False
        return getattr(backbones, model)(*a, **kw)

    timm.create_model = local_create
    torch.hub.load = local_hub
    repo = Path(r"D:\RC\lege\FoundationStereo")
    sys.path.insert(0, str(repo))
    from core.foundation_stereo import FoundationStereo
    from core.utils.utils import InputPadder
    checkpoint = repo / "pretrained_models/23-51-11/model_best_bp2.pth"
    cfg = OmegaConf.load(checkpoint.with_name("cfg.yaml"))
    cfg.vit_size, cfg.valid_iters = "vitl", 32
    print("Loading standard checkpoint on", torch.cuda.get_device_name(0), flush=True)
    model = FoundationStereo(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    del state
    print("Strict checkpoint load passed", flush=True)
    model = model.cuda().eval()
    packages = {n: importlib.metadata.version(n) for n in (
        "torch", "torchvision", "timm", "numpy", "opencv-python", "omegaconf")}
    metadata = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "strict_checkpoint_load_passed": True,
        "attention_backend": "pytorch_sdpa_compatibility",
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0),
                        "packages": packages},
    }
    return model, InputPadder, torch, metadata
