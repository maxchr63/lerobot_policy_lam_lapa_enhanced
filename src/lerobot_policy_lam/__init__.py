"""Installable plain LAM policy plugin for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("lerobot must be installed before importing lerobot_policy_lam.") from exc

from lerobot_policy_lam.configuration_lam import LAMConfig
from lerobot_policy_lam.modeling_lam import LAMPolicy
from lerobot_policy_lam.processor_lam import make_lam_pre_post_processors

__all__ = [
    "LAMConfig",
    "LAMPolicy",
    "make_lam_pre_post_processors",
]
