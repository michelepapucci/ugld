"""
UGLD: Uncertainty-Gated Lexical Decoding

Public API:
- UGLD_Towards, UGLD_Against (HuggingFace LogitsProcessor)
- UGLDTowardsConfig, UGLDAgainstConfig (configuration dataclasses)
"""

from .ugld import (
    UGLD_Towards,
    UGLD_Against,
    UGLDTowardsConfig,
    UGLDAgainstConfig,
)

__all__ = [
    "UGLD_Towards",
    "UGLD_Against",
    "UGLDTowardsConfig",
    "UGLDAgainstConfig",
]

__version__ = "1.0.0"
