#!/usr/bin/env python
"""Export offline LAM latent features for a LeRobot dataset.

This script computes latent features from a trained LAM checkpoint and writes
them to NumPy arrays. The current model can export:

- discrete code IDs
- continuous latent deltas in the quantizer embedding space
- hard codebook vectors in the quantizer embedding space

The final "add these arrays back into the dataset" step depends on the upstream
LeRobot dataset-editing feature proposed in:

https://github.com/huggingface/lerobot/pull/3136

Until that PR lands, this script intentionally stops at exporting:

- `<output_dir>/<feature_name>.npy`
- `<output_dir>/<valid_feature_name>.npy`

It also prints example `lerobot-edit-dataset --operation.type add_feature`
commands that can be run once PR #3136 is available in the user's `lerobot`
installation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot_policy_lam_lapa.modeling_lam import LAMPolicy

LATENT_FORMAT_CONTINUOUS = "continuous"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path",
        required=True,
        help="Path to a trained LAM checkpoint directory that can be loaded with LAMPolicy.from_pretrained().",
    )
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo_id.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional local dataset root. If omitted, LeRobot resolves the dataset as usual.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the exported .npy files will be written.",
    )
    parser.add_argument(
        "--camera-key",
        default=None,
        help="Camera feature to use. Defaults to the policy checkpoint camera key or the first dataset camera.",
    )
    parser.add_argument(
        "--future-frames",
        type=int,
        default=None,
        help="Override the policy checkpoint future_frames when pairing frames.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Number of frame pairs per forward pass.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override. Defaults to the checkpoint device or cuda if available.",
    )
    parser.add_argument(
        "--feature-name",
        default="latent_code",
        help="Name of the primary latent feature to export.",
    )
    parser.add_argument(
        "--latent-format",
        default=LATENT_FORMAT_CONTINUOUS,
        choices=[LATENT_FORMAT_CONTINUOUS],
        help="Which latent representation to export. Only continuous deltas are supported.",
    )
    parser.add_argument(
        "--valid-feature-name",
        default="latent_code_valid",
        help="Name of the companion validity-mask feature to export.",
    )
    parser.add_argument(
        "--new-repo-id",
        default=None,
        help="Optional repo_id for the post-export `lerobot-edit-dataset` example commands.",
    )
    parser.add_argument(
        "--new-root",
        default=None,
        help="Optional output root for the post-export `lerobot-edit-dataset` example commands.",
    )
    return parser.parse_args()


def _resolve_device(requested_device: str | None, policy: LAMPolicy) -> str:
    if requested_device is not None:
        return requested_device
    config_device = getattr(policy.config, "device", None)
    if config_device:
        return str(config_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_camera_key(requested_camera_key: str | None, policy: LAMPolicy, dataset: LeRobotDataset) -> str:
    if requested_camera_key:
        return requested_camera_key

    config_camera_key = getattr(policy.config, "camera_key", None)
    if config_camera_key:
        return str(config_camera_key)

    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        raise ValueError("Dataset has no camera keys. LAM export requires a visual input feature.")
    return str(camera_keys[0])


def _ensure_hwc_or_chw(image: object) -> torch.Tensor:
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dimensions, got shape {tuple(tensor.shape)}.")

    if tensor.shape[0] == 3:
        return tensor
    if tensor.shape[-1] == 3:
        return tensor.permute(2, 0, 1)

    raise ValueError(f"Unsupported image layout {tuple(tensor.shape)}. Expected CHW or HWC with 3 channels.")


def _load_frame_pair(dataset: LeRobotDataset, first_idx: int, second_idx: int, camera_key: str) -> torch.Tensor:
    first_item = dataset[first_idx]
    second_item = dataset[second_idx]
    first_image = _ensure_hwc_or_chw(first_item[camera_key])
    second_image = _ensure_hwc_or_chw(second_item[camera_key])
    return torch.stack((first_image, second_image), dim=0)


def _build_valid_pairs(dataset: LeRobotDataset, future_frames: int) -> tuple[np.ndarray, np.ndarray]:
    dataset._ensure_hf_dataset_loaded()
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    num_frames = len(episode_indices)
    valid_mask = np.zeros(num_frames, dtype=np.uint8)
    pair_targets = np.full(num_frames, -1, dtype=np.int64)

    if future_frames < 1:
        raise ValueError(f"future_frames must be >= 1, got {future_frames}.")

    for idx in range(num_frames - future_frames):
        target_idx = idx + future_frames
        if episode_indices[idx] != episode_indices[target_idx]:
            continue
        valid_mask[idx] = 1
        pair_targets[idx] = target_idx

    return valid_mask, pair_targets


def _get_latent_export_spec(policy: LAMPolicy, latent_format: str) -> tuple[tuple[int, ...], np.dtype, float | int, str]:
    code_seq_len = int(policy.config.code_seq_len)
    quant_dim = int(policy.config.quant_dim)

    if latent_format == LATENT_FORMAT_CONTINUOUS:
        return (code_seq_len, quant_dim), np.dtype(np.float32), 0.0, "float32"

    raise ValueError(f"Unsupported latent_format={latent_format!r}.")


def _extract_latents(policy: LAMPolicy, video: torch.Tensor, latent_format: str) -> torch.Tensor:
    return policy.extract_latents_from_video(video, latent_format=latent_format)


def _run_export(
    *,
    dataset: LeRobotDataset,
    policy: LAMPolicy,
    camera_key: str,
    future_frames: int,
    batch_size: int,
    latent_format: str,
    feature_name: str,
    valid_feature_name: str,
    output_dir: Path,
) -> tuple[Path, Path, list[int], str]:
    valid_mask, pair_targets = _build_valid_pairs(dataset, future_frames=future_frames)
    num_frames = int(valid_mask.shape[0])
    latent_shape_tail, latent_dtype, invalid_fill_value, feature_dtype = _get_latent_export_spec(
        policy,
        latent_format=latent_format,
    )

    latents = np.full((num_frames, *latent_shape_tail), invalid_fill_value, dtype=latent_dtype)
    valid_indices = np.flatnonzero(valid_mask)

    logging.info(
        "Exporting %d valid frame pairs out of %d total frames using camera_key=%s, future_frames=%d, latent_format=%s.",
        len(valid_indices),
        num_frames,
        camera_key,
        future_frames,
        latent_format,
    )

    if len(valid_indices) > 0:
        policy.eval()
        with torch.inference_mode():
            for start in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[start : start + batch_size]
                frame_pairs = [
                    _load_frame_pair(
                        dataset=dataset,
                        first_idx=int(frame_idx),
                        second_idx=int(pair_targets[frame_idx]),
                        camera_key=camera_key,
                    )
                    for frame_idx in batch_indices
                ]
                batch = {camera_key: torch.stack(frame_pairs, dim=0)}
                video, valid_pair, _ = policy._extract_frame_pair(batch)
                if not bool(valid_pair.all().item()):
                    raise RuntimeError("Unexpected invalid pair in export batch.")
                latent_values = _extract_latents(policy=policy, video=video, latent_format=latent_format)
                latents[batch_indices] = latent_values.detach().to("cpu").numpy().astype(latent_dtype, copy=False)
                logging.info("Processed %d / %d valid frame pairs.", min(start + batch_size, len(valid_indices)), len(valid_indices))

    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = output_dir / f"{feature_name}.npy"
    valid_path = output_dir / f"{valid_feature_name}.npy"
    np.save(latent_path, latents, allow_pickle=False)
    np.save(valid_path, valid_mask.reshape(num_frames, 1), allow_pickle=False)
    return latent_path, valid_path, list(latent_shape_tail), feature_dtype


def _format_add_feature_command(
    *,
    repo_id: str,
    root: str | None,
    new_repo_id: str | None,
    new_root: str | None,
    feature_name: str,
    feature_values_path: Path,
    feature_dtype: str,
    feature_shape: list[int],
) -> str:
    parts = [
        "lerobot-edit-dataset",
        f"--repo_id {repo_id}",
    ]
    if root:
        parts.append(f"--root {root}")
    if new_repo_id:
        parts.append(f"--new_repo_id {new_repo_id}")
    if new_root:
        parts.append(f"--new_root {new_root}")
    parts.extend(
        [
            "--operation.type add_feature",
            f"--operation.feature_name {feature_name}",
            f"--operation.feature_values_path {feature_values_path}",
            f"--operation.feature_dtype {feature_dtype}",
            f"--operation.feature_shape \"{feature_shape}\"",
        ]
    )
    return " \\\n  ".join(parts)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    policy = LAMPolicy.from_pretrained(args.policy_path, local_files_only=True)
    device = _resolve_device(args.device, policy)
    policy.to(device)
    policy.config.device = device

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        download_videos=True,
    )

    camera_key = _resolve_camera_key(args.camera_key, policy, dataset)
    future_frames = int(args.future_frames if args.future_frames is not None else policy.config.future_frames)
    output_dir = Path(args.output_dir)

    latent_path, valid_path, latent_feature_shape, latent_feature_dtype = _run_export(
        dataset=dataset,
        policy=policy,
        camera_key=camera_key,
        future_frames=future_frames,
        batch_size=args.batch_size,
        latent_format=args.latent_format,
        feature_name=args.feature_name,
        valid_feature_name=args.valid_feature_name,
        output_dir=output_dir,
    )

    logging.info("Wrote latent codes to %s", latent_path)
    logging.info("Wrote validity mask to %s", valid_path)
    logging.info("Post-export dataset editing depends on https://github.com/huggingface/lerobot/pull/3136")

    print()
    print("After huggingface/lerobot#3136 lands, these arrays can be added back into the dataset with:")
    print()
    print(
        _format_add_feature_command(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            new_repo_id=args.new_repo_id,
            new_root=args.new_root,
            feature_name=args.feature_name,
            feature_values_path=latent_path,
            feature_dtype=latent_feature_dtype,
            feature_shape=latent_feature_shape,
        )
    )
    print()
    print("Then add the companion validity mask with a second add_feature invocation:")
    print()
    print(
        _format_add_feature_command(
            repo_id=args.new_repo_id or args.dataset_repo_id,
            root=args.new_root or args.dataset_root,
            new_repo_id=None,
            new_root=None,
            feature_name=args.valid_feature_name,
            feature_values_path=valid_path,
            feature_dtype="uint8",
            feature_shape=[1],
        )
    )


if __name__ == "__main__":
    main()
