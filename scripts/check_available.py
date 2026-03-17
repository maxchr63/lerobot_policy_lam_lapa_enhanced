"""Smoke-check that the installed plugin registers and can run a forward pass."""

import logging
from types import SimpleNamespace

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins


logging.basicConfig(level=logging.INFO)

register_third_party_plugins()

LAMConfig = PreTrainedConfig.get_choice_class("lam")
config = LAMConfig(
    device="cpu",
    future_frames=1,
    image_size=(128, 128),
    patch_size=(16, 16),
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
dataset_meta = SimpleNamespace(
    features={
        "observation.images.image": {
            "dtype": "video",
            "shape": [128, 128, 3],
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["action"],
        },
    },
    stats={},
)

policy = make_policy(config, ds_meta=dataset_meta)
make_pre_post_processors(config)

batch = {
    "observation.images.image": torch.randint(0, 255, (2, 2, 3, 128, 128), dtype=torch.uint8),
    "observation.images.image_is_pad": torch.tensor([[False, False], [False, True]]),
    "action": torch.randn(2, 7),
}
loss, output_dict = policy.forward(batch)
logging.info("loss=%s output=%s", float(loss.item()), output_dict)
