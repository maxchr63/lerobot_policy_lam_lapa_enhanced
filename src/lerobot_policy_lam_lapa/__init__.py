"""Installable LAM LAPA policy plugin for LeRobot with a DINOv3 encoder frontend."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:
    raise ImportError("lerobot must be installed before importing lerobot_policy_lam_lapa.") from exc

from lerobot_policy_lam_lapa.configuration_lam import LAMConfig
from lerobot_policy_lam_lapa.modeling_lam import LAMPolicy
from lerobot_policy_lam_lapa.processor_lam import make_lam_lapa_pre_post_processors

__all__ = [
    "LAMConfig",
    "LAMPolicy",
    "make_lam_lapa_pre_post_processors",
]
