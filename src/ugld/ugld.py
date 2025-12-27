# ugld.py
# Uncertainty-Gated Lexical Decoding (UGLD)
#
# This file implements the two variants described in the paper:
# - UGLD-t (towards): mix the model distribution p with a prior q supported only on green tokens
#   p' = (1-α) p + α q, where α = α_max * φ(p)
# - UGLD-a (against): penalize red tokens in logit space
#   z' = z - λ r, where λ = λ_max * φ(p)
#
# The uncertainty gate is computed from Shannon entropy:
#   H(p) = - Σ_i p_i log p_i
#   φ(p) = σ((H(p) - τ) / s)
#
# Notes for library use:
# - Both classes are HuggingFace LogitsProcessor and can be passed to LogitsProcessorList.
# - Both are batch-safe: input logits are [B, V].
# - We keep everything torch.no_grad() for efficiency in decoding.
# - We validate token ids by clipping to [0, V).

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import torch
from transformers import LogitsProcessor


def _entropy_from_probs(p: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Shannon entropy per batch row.
    p: [B, V] probability simplex
    returns: H: [B]
    """
    return -(p * p.clamp_min(eps).log()).sum(dim=-1)


def _valid_token_ids(ids: Sequence[int] | Iterable[int], V: int, device: torch.device) -> torch.Tensor:
    """
    Convert ids to a unique sorted LongTensor on device, filtered to [0, V).
    """
    uniq = sorted({int(i) for i in ids})
    if not uniq:
        return torch.empty((0,), dtype=torch.long, device=device)
    t = torch.tensor(uniq, dtype=torch.long, device=device)
    return t[(t >= 0) & (t < V)]


def _gate_from_entropy(H: torch.Tensor, tau: float, s: float) -> torch.Tensor:
    """
    φ(p) = σ((H-τ)/s), with s strictly positive.
    """
    if s <= 0:
        raise ValueError("UGLD requires s > 0.")
    return torch.sigmoid((H - tau) / s)


@dataclass(frozen=True)
class UGLDTowardsConfig:
    green_token_ids: Sequence[int]
    alpha_max: float = 0.25
    tau: float = 3.0
    s: float = 0.3
    eps: float = 1e-12
    prior: Literal["uniform", "topk", "renorm"] = "renorm"
    topk: int = 16


class UGLD_Towards(LogitsProcessor):
    """
    UGLD-t: condition *towards* a set of tokens (green vocabulary).

    At each decoding step:
      p = SoftMax(z)
      H = -Σ p log p
      φ = σ((H-τ)/s)
      α = α_max * φ
      p' = (1-α)p + α q

    q is a prior supported only on green tokens, with three options:
      - "uniform": uniform over all valid green tokens
      - "topk": uniform over the top-K green tokens by current p
      - "renorm": q_i = p_i / Σ_{j in G} p_j  for i in G; else 0
    """

    def __init__(self, config: UGLDTowardsConfig):
        super().__init__()
        if not (0.0 <= config.alpha_max <= 1.0):
            raise ValueError("alpha_max must be in [0, 1].")
        if config.topk <= 0:
            raise ValueError("topk must be > 0.")
        self.cfg = config

        # Cache for uniform q (depends on vocab size/device/dtype).
        self._uniform_q: Optional[torch.Tensor] = None
        self._uniform_meta = None  # (V, device, dtype)

    def _uniform_prior(self, V: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        q: [V] uniform over green tokens, else 0. Cached.
        """
        meta = (V, device, dtype)
        if self._uniform_q is not None and self._uniform_meta == meta:
            return self._uniform_q

        green = _valid_token_ids(self.cfg.green_token_ids, V, device)
        q = torch.zeros((V,), dtype=dtype, device=device)
        if green.numel() > 0:
            q[green] = 1.0 / float(green.numel())

        self._uniform_q = q
        self._uniform_meta = meta
        return q

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        scores: [B, V] logits
        returns: [B, V] logits (we return log-probabilities for numerical stability;
                 SoftMax(log p') == p')
        """
        B, V = scores.shape
        device, dtype = scores.device, scores.dtype

        # Base distribution p over vocab.
        p = torch.softmax(scores, dim=-1)  # [B, V]

        # Entropy gate.
        H = _entropy_from_probs(p, self.cfg.eps)         # [B]
        phi = _gate_from_entropy(H, self.cfg.tau, self.cfg.s)  # [B] in [0,1]
        alpha = (self.cfg.alpha_max * phi).clamp(0.0, 1.0)     # [B]

        # Valid green token ids in this vocab.
        green = _valid_token_ids(self.cfg.green_token_ids, V, device)
        if green.numel() == 0:
            # No valid green tokens -> no-op.
            return scores

        # Build q: [B, V]
        if self.cfg.prior == "uniform":
            q = self._uniform_prior(V, device, dtype).unsqueeze(0).expand(B, V)

        else:
            q = torch.zeros_like(p)  # [B, V]
            mass_g = p.index_select(dim=-1, index=green)  # [B, |G|]

            if self.cfg.prior == "renorm":
                denom = mass_g.sum(dim=-1, keepdim=True).clamp_min(self.cfg.eps)
                q_g = mass_g / denom  # [B, |G|]

            elif self.cfg.prior == "topk":
                k = min(self.cfg.topk, green.numel())
                _, idx = torch.topk(mass_g, k=k, dim=-1)  # [B, k] indices into green
                chosen = green.index_select(0, idx.reshape(-1)).reshape(B, k)  # [B, k]
                # Uniform over chosen top-k.
                q.scatter_(
                    dim=-1,
                    index=chosen,
                    src=torch.full((B, k), 1.0 / float(k), device=device, dtype=dtype),
                )
                # Done.
                p_prime = (1.0 - alpha.unsqueeze(-1)) * p + alpha.unsqueeze(-1) * q
                return (p_prime.clamp_min(self.cfg.eps)).log()

            else:
                raise ValueError(f"Unknown prior='{self.cfg.prior}'. Use: uniform|topk|renorm")

            # Scatter q_g back into vocab positions.
            q.scatter_(dim=-1, index=green.unsqueeze(0).expand(B, -1), src=q_g)

        # Mix in probability space (convex combination).
        p_prime = (1.0 - alpha.unsqueeze(-1)) * p + alpha.unsqueeze(-1) * q
        return (p_prime.clamp_min(self.cfg.eps)).log()


@dataclass(frozen=True)
class UGLDAgainstConfig:
    red_token_ids: Sequence[int]
    lambda_max: float = 4.0
    tau: float = 3.0
    s: float = 0.3
    eps: float = 1e-12
    weights: Literal["fixed", "dynamic_minmax"] = "fixed"
    fixed_r: float = 1.0


class UGLD_Against(LogitsProcessor):
    """
    UGLD-a: condition *against* a set of tokens (red vocabulary) in logit space.

    At each decoding step:
      p = SoftMax(z)
      H = -Σ p log p
      φ = σ((H-τ)/s)
      λ = λ_max * φ
      z' = z - λ r

    r is a non-negative weight vector supported only on red tokens:
      - "fixed": r_i = fixed_r for i in R; else 0
      - "dynamic_minmax": allocate larger penalties to red tokens the model currently
        prefers, using min-max normalization of {p_i : i in R} mapped to [1,2]:
           f_i = (p_i - min(p^R)) / (max(p^R) - min(p^R) + eps)
           r_i = 1 + f_i   for i in R; else 0
    """

    def __init__(self, config: UGLDAgainstConfig):
        super().__init__()
        if config.lambda_max < 0:
            raise ValueError("lambda_max must be >= 0.")
        if config.fixed_r <= 0:
            raise ValueError("fixed_r must be > 0 (weights must be positive on red tokens).")
        self.cfg = config

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        scores: [B, V] logits
        returns: [B, V] logits after applying a gated penalty on red tokens
        """
        B, V = scores.shape
        device = scores.device

        red = _valid_token_ids(self.cfg.red_token_ids, V, device)
        if red.numel() == 0 or self.cfg.lambda_max == 0.0:
            return scores

        # Compute entropy gate from current model distribution p.
        p = torch.softmax(scores, dim=-1)  # [B, V]
        H = _entropy_from_probs(p, self.cfg.eps)  # [B]
        phi = _gate_from_entropy(H, self.cfg.tau, self.cfg.s)  # [B]
        lam = (self.cfg.lambda_max * phi).clamp_min(0.0)  # [B]

        # Build r: [B, V]
        r = torch.zeros_like(scores)

        if self.cfg.weights == "fixed":
            r[:, red] = self.cfg.fixed_r

        elif self.cfg.weights == "dynamic_minmax":
            pr = p.index_select(dim=-1, index=red)  # [B, |R|]
            pr_min = pr.min(dim=-1, keepdim=True).values
            pr_max = pr.max(dim=-1, keepdim=True).values
            f = (pr - pr_min) / (pr_max - pr_min + self.cfg.eps)  # [B, |R|] in [0,1]
            r_r = 1.0 + f  # [B, |R|] in [1,2]
            r.scatter_(dim=-1, index=red.unsqueeze(0).expand(B, -1), src=r_r)

        else:
            raise ValueError(f"Unknown weights='{self.cfg.weights}'. Use: fixed|dynamic_minmax")

        # Penalize in logit space: z' = z - λ r
        return scores - lam.unsqueeze(-1) * r
