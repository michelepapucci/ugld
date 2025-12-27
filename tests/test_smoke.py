import torch

from ugld import (
    UGLD_Towards,
    UGLD_Against,
    UGLDTowardsConfig,
    UGLDAgainstConfig,
)


def test_ugld_towards_runs():
    """
    Smoke test: UGLD-t should run and preserve shape.
    """
    B, V = 2, 50
    scores = torch.randn(B, V)

    processor = UGLD_Towards(
        UGLDTowardsConfig(
            green_token_ids=[1, 2, 3],
            alpha_max=0.5,
            tau=1.0,
            s=0.3,
            prior="renorm",
        )
    )

    out = processor(
        input_ids=torch.zeros(B, 1, dtype=torch.long),
        scores=scores.clone(),
    )

    assert out.shape == scores.shape
    assert torch.isfinite(out).all()


def test_ugld_against_runs():
    """
    Smoke test: UGLD-a should run and preserve shape.
    """
    B, V = 2, 50
    scores = torch.randn(B, V)

    processor = UGLD_Against(
        UGLDAgainstConfig(
            red_token_ids=[4, 5, 6],
            lambda_max=2.0,
            tau=1.0,
            s=0.3,
            weights="fixed",
        )
    )

    out = processor(
        input_ids=torch.zeros(B, 1, dtype=torch.long),
        scores=scores.clone(),
    )

    assert out.shape == scores.shape
    assert torch.isfinite(out).all()
