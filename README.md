# LeRobot LAM Policy

Installable third-party LeRobot policy package that adds a plain latent action model
(LAM) policy under `policy.type=lam`.

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

## Notes

- `policy.type` is `lam`.
- This is a training-only policy. `select_action` is intentionally unsupported.
- The dataset `action` field is still present in standard LeRobot batches, but it is
  ignored by the LAM loss.
- The temporal offset is controlled by `policy.future_frames`.
