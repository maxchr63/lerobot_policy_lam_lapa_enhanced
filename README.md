# LeRobot LAM LAPA Policy

Installable third-party LeRobot policy package that adds a plain latent action model
(LAM) policy under `policy.type=lam_lapa`.

The installable package name is `lerobot_policy_lam_lapa`.

This package is designed to be discovered by `lerobot` via
`lerobot.utils.import_utils.register_third_party_plugins()`.

## Install

```bash
conda run -n lerobot pip install -e .
```

## Smoke check

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
  --steps=200 \
```

## Notes

- `policy.type` is `lam_lapa`.
- This is a training-only policy. `select_action` is intentionally unsupported.
- The dataset `action` field is still present in standard LeRobot batches, but it is
  ignored by the LAM loss.
- The temporal offset is controlled by `policy.future_frames`.
