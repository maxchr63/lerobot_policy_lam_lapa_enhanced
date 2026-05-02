"""Smoke-check that the installed plugin registers and can run a forward pass.

Tests:
  1. Single-camera forward pass (regression — must match pre-multicam behaviour)
  2. Two-camera with default fusion_mode='spatial_64' (spatially grounded)
  3. Three-camera with view dropout (default 'spatial_64')
  4. Two-camera with fusion_mode='spatial_4'
  5. Two-camera with fusion_mode='pool_4' (cheapest baseline)
  6. Two-camera with view dropout that zeroes ALL extras for some samples
     (validates the all-extras-dropped safety guard)
"""

import logging
import sys
from types import SimpleNamespace

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

register_third_party_plugins()

LAMConfig = PreTrainedConfig.get_choice_class("lam_lapa")

_TINY = dict(
    device="cpu",
    future_frames=1,
    image_size=(256, 256),
    dim=64,
    quant_dim=8,
    codebook_size=16,
    code_seq_len=1,
    spatial_depth=1,
    temporal_depth=1,
    dim_head=16,
    heads=2,
    channels=3,
    metrics_num_unique_codes_every_n_steps=1,
)

_FEATURES_1CAM = {
    "observation.images.image": {
        "dtype": "video", "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
    },
    "action": {"dtype": "float32", "shape": [7], "names": ["action"]},
}

_FEATURES_2CAM = {
    "observation.images.top":   {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.images.wrist": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "action": {"dtype": "float32", "shape": [7], "names": ["action"]},
}

_FEATURES_3CAM = {
    "observation.images.top":   {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.images.wrist": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "observation.images.side":  {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
    "action": {"dtype": "float32", "shape": [7], "names": ["action"]},
}


def _rand_frames(B=2):
    return torch.randint(0, 255, (B, 2, 3, 256, 256), dtype=torch.uint8)


def _make_2cam_config(mode: str, view_dropout: float = 0.0):
    return LAMConfig(
        **_TINY,
        camera_keys=["observation.images.top", "observation.images.wrist"],
        camera_key_to_slot={"observation.images.top": 0, "observation.images.wrist": 1},
        n_bottleneck_tokens=4,
        bottleneck_heads=2,
        view_dropout_prob=view_dropout,
        fusion_mode=mode,
    )


# ── Test 1: single-camera (unchanged path) ───────────────────────────────────
logging.info("=== Test 1: single-camera ===")
config = LAMConfig(**_TINY)
ds_meta = SimpleNamespace(features=_FEATURES_1CAM, stats={})
policy = make_policy(config, ds_meta=ds_meta)
make_pre_post_processors(config)

assert policy.lam.bottleneck_fusion is None, "no bottleneck_fusion for single-cam"
assert policy.lam.spatial_cross is None, "no spatial_cross for single-cam"

batch = {
    "observation.images.image": _rand_frames(),
    "observation.images.image_is_pad": torch.tensor([[False, False], [False, True]]),
    "action": torch.randn(2, 7),
}
loss, metrics = policy.forward(batch)
assert "fusion_attn_cam0_frame_t" not in metrics, "no fusion metrics in single-cam"
logging.info("PASS — loss=%.4f", float(loss.item()))


# ── Test 2: two-camera with default fusion_mode='spatial_64' ─────────────────
logging.info("=== Test 2: two-camera (default fusion_mode='spatial_64') ===")
config2 = _make_2cam_config(mode="spatial_64", view_dropout=0.2)
assert config2.fusion_mode == "spatial_64", \
    f"default mode must be 'spatial_64', got {config2.fusion_mode}"
ds_meta2 = SimpleNamespace(features=_FEATURES_2CAM, stats={})
policy2 = make_policy(config2, ds_meta=ds_meta2)
policy2.train()

assert policy2.lam.spatial_cross is not None, "spatial_cross must be built"
assert policy2.lam.bottleneck_fusion is None, "bottleneck_fusion must NOT be built"
assert policy2.lam.view_id_embedding.weight.shape[0] == 8, "embedding table size 8"
# Gate must initialise to 0 → fusion contributes 0 at step 0.
assert torch.allclose(policy2.lam.spatial_cross.gate, torch.zeros(1)), \
    "spatial_cross gate must init to 0"

batch2 = {
    "observation.images.top":   _rand_frames(),
    "observation.images.wrist": _rand_frames(),
    "action": torch.randn(2, 7),
}
loss2, metrics2 = policy2.forward(batch2)
# spatial_64 defaults to fusion_keys_include_primary=False — primary patches
# are already queries, including them as keys is redundant self-attention.
assert config2.fusion_keys_include_primary is False, \
    "spatial_64 default for fusion_keys_include_primary must be False"
assert "fusion_attn_cam0_frame_t" in metrics2
assert "fusion_attn_cam0_frame_t1" in metrics2
assert "fusion_attn_cam1_frame_t" in metrics2
assert "fusion_attn_cam1_frame_t1" in metrics2
# Primary slot is hard-zero when excluded.
assert metrics2["fusion_attn_cam0_frame_t"]  == 0.0
assert metrics2["fusion_attn_cam0_frame_t1"] == 0.0
logging.info("PASS — loss=%.4f attn frame_t=[cam0=%.3f cam1=%.3f] frame_t1=[cam0=%.3f cam1=%.3f] (spatial_64, primary excluded)",
             float(loss2.item()),
             metrics2["fusion_attn_cam0_frame_t"],  metrics2["fusion_attn_cam1_frame_t"],
             metrics2["fusion_attn_cam0_frame_t1"], metrics2["fusion_attn_cam1_frame_t1"])


# ── Test 3: three-camera with view dropout ──────────────────────────────────
logging.info("=== Test 3: three-camera (default 'spatial_64', view_dropout=0.5) ===")
config3 = LAMConfig(
    **_TINY,
    camera_keys=["observation.images.top", "observation.images.wrist", "observation.images.side"],
    n_bottleneck_tokens=4,
    bottleneck_heads=2,
    view_dropout_prob=0.5,
)
ds_meta3 = SimpleNamespace(features=_FEATURES_3CAM, stats={})
policy3 = make_policy(config3, ds_meta=ds_meta3)
policy3.train()

batch3 = {
    "observation.images.top":   _rand_frames(),
    "observation.images.wrist": _rand_frames(),
    "observation.images.side":  _rand_frames(),
    "action": torch.randn(2, 7),
}
loss3, metrics3 = policy3.forward(batch3)
assert "fusion_attn_cam2_frame_t" in metrics3
assert "fusion_attn_cam2_frame_t1" in metrics3
logging.info("PASS — loss=%.4f frame_t=[cam0=%.3f cam1=%.3f cam2=%.3f]",
             float(loss3.item()),
             metrics3["fusion_attn_cam0_frame_t"],
             metrics3["fusion_attn_cam1_frame_t"],
             metrics3["fusion_attn_cam2_frame_t"])


# ── Test 4: two-camera fusion_mode='spatial_4' ──────────────────────────────
logging.info("=== Test 4: two-camera fusion_mode='spatial_4' ===")
config4 = _make_2cam_config(mode="spatial_4")
ds_meta4 = SimpleNamespace(features=_FEATURES_2CAM, stats={})
policy4 = make_policy(config4, ds_meta=ds_meta4)
policy4.train()
assert policy4.lam.bottleneck_fusion is not None
assert policy4.lam.spatial_cross is None

loss4, metrics4 = policy4.forward(batch2)
# spatial_4 default: fusion_keys_include_primary=True
assert config4.fusion_keys_include_primary is True
logging.info("PASS — loss=%.4f frame_t=[cam0=%.3f cam1=%.3f] frame_t1=[cam0=%.3f cam1=%.3f] (spatial_4)",
             float(loss4.item()),
             metrics4["fusion_attn_cam0_frame_t"],  metrics4["fusion_attn_cam1_frame_t"],
             metrics4["fusion_attn_cam0_frame_t1"], metrics4["fusion_attn_cam1_frame_t1"])


# ── Test 5: two-camera fusion_mode='pool_4' (baseline) ──────────────────────
logging.info("=== Test 5: two-camera fusion_mode='pool_4' (cheapest baseline) ===")
config5 = _make_2cam_config(mode="pool_4")
ds_meta5 = SimpleNamespace(features=_FEATURES_2CAM, stats={})
policy5 = make_policy(config5, ds_meta=ds_meta5)
policy5.train()

loss5, metrics5 = policy5.forward(batch2)
# pool_4 default: fusion_keys_include_primary=True → 2 keys, non-degenerate softmax
assert config5.fusion_keys_include_primary is True
logging.info("PASS — loss=%.4f frame_t=[cam0=%.3f cam1=%.3f] frame_t1=[cam0=%.3f cam1=%.3f] (pool_4)",
             float(loss5.item()),
             metrics5["fusion_attn_cam0_frame_t"],  metrics5["fusion_attn_cam1_frame_t"],
             metrics5["fusion_attn_cam0_frame_t1"], metrics5["fusion_attn_cam1_frame_t1"])


# ── Test 6: all-extras-dropped safety guard ─────────────────────────────────
# Simulate the catastrophic case where every sample has all extras dropped.
# We do this by constructing a present_mask manually and calling the encoder
# directly (the public training path randomises dropout per step).
logging.info("=== Test 6: all-extras-dropped safety guard ===")
B = 2
device = "cpu"
fake_pairs = [
    (torch.randn(B, 3, 1, 256, 256), torch.randn(B, 3, 1, 256, 256)),
    (torch.randn(B, 3, 1, 256, 256), torch.randn(B, 3, 1, 256, 256)),
]
# Every sample has primary on, extras off.
all_dropped_mask = torch.tensor([[True, False], [True, False]])
policy2.eval()
with torch.no_grad():
    out = policy2.lam._encode_frames_multi(fake_pairs, present_mask=all_dropped_mask)
# Should not produce NaN.
assert not any(torch.isnan(t).any().item() for t in out[:4]), \
    "all-extras-dropped produced NaN — safety guard failed"
logging.info("PASS — no NaN with all-extras-dropped present_mask")



# ── Test 7: legacy extras-only baseline (fusion_keys_include_primary=False) ─
logging.info("=== Test 7: pool_4 with fusion_keys_include_primary=False (legacy) ===")
config7 = LAMConfig(
    **_TINY,
    camera_keys=["observation.images.top", "observation.images.wrist"],
    camera_key_to_slot={"observation.images.top": 0, "observation.images.wrist": 1},
    n_bottleneck_tokens=4,
    bottleneck_heads=2,
    view_dropout_prob=0.0,
    fusion_mode="pool_4",
    fusion_keys_include_primary=False,
)
ds_meta7 = SimpleNamespace(features=_FEATURES_2CAM, stats={})
policy7 = make_policy(config7, ds_meta=ds_meta7)
policy7.train()

loss7, metrics7 = policy7.forward(batch2)
# In legacy mode primary is excluded — its slot is hard-zero in both frames.
assert metrics7["fusion_attn_cam0_frame_t"]  == 0.0
assert metrics7["fusion_attn_cam0_frame_t1"] == 0.0
# With N=2 and primary excluded, only cam1 is in keys → attn=1.0 (degenerate
# softmax over 1 key, exactly the issue this change fixes by default).
assert abs(metrics7["fusion_attn_cam1_frame_t"]  - 1.0) < 1e-5
assert abs(metrics7["fusion_attn_cam1_frame_t1"] - 1.0) < 1e-5
logging.info("PASS — loss=%.4f frame_t=[cam0=%.3f cam1=%.3f] frame_t1=[cam0=%.3f cam1=%.3f] (legacy extras-only)",
             float(loss7.item()),
             metrics7["fusion_attn_cam0_frame_t"],  metrics7["fusion_attn_cam1_frame_t"],
             metrics7["fusion_attn_cam0_frame_t1"], metrics7["fusion_attn_cam1_frame_t1"])


logging.info("=== All smoke tests passed ===")
