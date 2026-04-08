# LeRobot LAM LAPA Policy

Installable third-party LeRobot policy package that adds a standalone latent action
model policy under `policy.type=lam_lapa`.

The installable package name is `lerobot_policy_lam_lapa`.

This package is designed to be discovered by `lerobot` via
`lerobot.utils.import_utils.register_third_party_plugins()`.

## Architecture

- Encoder frontend: frozen `facebook/dinov3-vits16-pretrain-lvd1689m`
- Learned downsampler: `16x16x384 -> 8x8xdim` via stride-2 `2x2` convolution
- Decoder context: separate detached pixel-context projection
- Core LAPA components: spatial transformer, temporal transformer, NSVQ, pixel decoder

The current implementation is intentionally narrow:

- `image_size=(256, 256)` only
- DINOv3-S is the default backbone
- Decoder context is always detached pixel context

## Install

```bash
conda run -n lerobot pip install -e .
```

## Smoke check

This downloads the DINOv3-S backbone the first time it runs.

```bash
conda run -n lerobot python scripts/check_available.py
```

## Example train command

This command was validated against `HuggingFaceVLA/libero` in the local
`lerobot` environment.

```bash
lerobot-train \
  --policy.push_to_hub=false \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.type=lam_lapa \
  --policy.camera_key=observation.images.image \
  --policy.future_frames=10 \
  --batch_size=8 \
  --steps=200
```

## Notes

- `policy.type` is `lam_lapa`.
- This is a training-only policy. `select_action` is intentionally unsupported.
- The latent export script is available at `scripts/export_lam_latents.py`.
