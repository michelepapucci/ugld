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
    """Compute Shannon entropy for each row of a probability matrix.

    Args:
        p: Probability tensor of shape ``[B, V]``, where each row is a valid
            probability simplex over the vocabulary.
        eps: Small constant to clamp probabilities before taking the log,
            preventing ``log(0)``.

    Returns:
        Entropy tensor of shape ``[B]``, where entry *b* is
        ``H(p[b]) = -sum_i p[b,i] * log(p[b,i])``.
    """
    return -(p * p.clamp_min(eps).log()).sum(dim=-1)


def _valid_token_ids(ids: Sequence[int] | Iterable[int], V: int, device: torch.device) -> torch.Tensor:
    """Sanitise a collection of token ids for use against a vocabulary of size *V*.

    Deduplicates and sorts the ids, then discards any that fall outside
    ``[0, V)``.

    Args:
        ids: Iterable of integer token ids (may contain duplicates or
            out-of-range values).
        V: Vocabulary size; only ids in ``[0, V)`` are kept.
        device: Target device for the returned tensor.

    Returns:
        1-D ``LongTensor`` of unique, sorted, in-range token ids on *device*.
        Returns an empty tensor if no valid ids remain.
    """
    uniq = sorted({int(i) for i in ids})
    if not uniq:
        return torch.empty((0,), dtype=torch.long, device=device)
    t = torch.tensor(uniq, dtype=torch.long, device=device)
    return t[(t >= 0) & (t < V)]


def _gate_from_entropy(H: torch.Tensor, tau: float, s: float) -> torch.Tensor:
    """Compute the uncertainty gate φ(p) = σ((H − τ) / s).

    The gate is close to 0 when entropy is well below the threshold *τ*
    (model is confident) and close to 1 when entropy is well above *τ*
    (model is uncertain).  The smoothing factor *s* controls how sharply
    the gate transitions between the two regimes.

    Args:
        H: Per-sample entropy tensor of shape ``[B]``.
        tau: Entropy threshold.  The gate reaches 0.5 when ``H == tau``.
            Should be chosen in ``[0, log |V|]``; values outside this range
            produce a gate that is always closed or always open.
        s: Smoothing factor (must be strictly positive).  Smaller values
            make the gate switch more abruptly.

    Returns:
        Gate tensor of shape ``[B]`` with values in ``(0, 1)``.

    Raises:
        ValueError: If *s* is not strictly positive.
    """
    if s <= 0:
        raise ValueError("UGLD requires s > 0.")
    return torch.sigmoid((H - tau) / s)


@dataclass(frozen=True)
class UGLDTowardsConfig:
    """Configuration for :class:`UGLD_Towards`.

    Attributes:
        green_token_ids: Token ids that form the *green* vocabulary — the set
            of tokens the model is encouraged to generate.  Duplicates and
            out-of-range ids are ignored at runtime.
        alpha_max: Maximum mixing coefficient α ∈ [0, 1].  The effective α at
            each step is ``alpha_max * φ(p)``, so the actual intervention is
            always at most *alpha_max*.  Defaults to ``0.25``.
        tau: Entropy threshold τ for the gate φ.  The gate is ~0.5 when the
            per-token entropy equals *tau*.  A good starting point is the
            median entropy over your dataset's decoding steps.  Defaults to
            ``3.0``.
        s: Smoothing factor s > 0 for the gate sigmoid.  Smaller values make
            the gate switch more sharply.  Defaults to ``0.3``.
        eps: Small constant for numerical stability in log and division
            operations.  Defaults to ``1e-12``.
        prior: Which conditioning prior *q* to use:

            - ``"uniform"`` — uniform mass over all green tokens.
            - ``"topk"`` — uniform mass over the *topk* green tokens with the
              highest probability under the current model distribution.
            - ``"renorm"`` — renormalise the current model distribution
              restricted to green tokens (i.e. ``q_i ∝ p_i`` for i ∈ G).

            Defaults to ``"renorm"``.
        topk: Number of green candidates to keep when ``prior="topk"``.
            Clamped to the number of valid green tokens at runtime.
            Defaults to ``16``.
    """

    green_token_ids: Sequence[int]
    alpha_max: float = 0.25
    tau: float = 3.0
    s: float = 0.3
    eps: float = 1e-12
    prior: Literal["uniform", "topk", "renorm"] = "renorm"
    topk: int = 16


class UGLD_Towards(LogitsProcessor):
    """Condition generation *towards* a predefined vocabulary (UGLD-t).

    At each decoding step the model's next-token distribution *p* is mixed
    with a conditioning prior *q* that concentrates probability mass on the
    *green* tokens.  The mixing strength is gated by the model's predictive
    uncertainty, measured via Shannon entropy, so that intervention is strong
    when the model is uncertain and negligible when it is confident.

    Formally, at each step:

    .. code-block:: text

        p  = SoftMax(z)               # current model distribution
        H  = -Σ p_i log p_i           # Shannon entropy
        φ  = σ((H - τ) / s)           # uncertainty gate ∈ (0, 1)
        α  = α_max · φ                # effective mixing coefficient
        p' = (1 − α) p + α q          # conditioned distribution

    The output is ``log(p')``; because ``SoftMax(log(p')) = p'``, this is a
    valid drop-in replacement for the raw logits expected by the HuggingFace
    generation pipeline.

    Args:
        config: A :class:`UGLDTowardsConfig` instance specifying the green
            vocabulary and all hyperparameters.

    Raises:
        ValueError: If ``config.alpha_max`` is outside ``[0, 1]`` or
            ``config.topk`` is not positive.

    Example::

        from transformers import LogitsProcessorList
        from ugld import UGLD_Towards, UGLDTowardsConfig

        processor = UGLD_Towards(UGLDTowardsConfig(
            green_token_ids=green_ids,
            alpha_max=0.5,
            tau=1.0,
            s=0.3,
            prior="renorm",
        ))
        out = model.generate(**inputs, logits_processor=LogitsProcessorList([processor]))
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
        """Build (and cache) a uniform prior over the green vocabulary.

        Returns a 1-D tensor of shape ``[V]`` where each green token receives
        probability ``1 / |G|`` and all other tokens receive ``0``.  The
        result is cached and reused as long as *V*, *device*, and *dtype*
        remain unchanged.

        Args:
            V: Vocabulary size.
            device: Target device.
            dtype: Target floating-point dtype.

        Returns:
            Prior tensor of shape ``[V]``.
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
        """Apply UGLD-t to a batch of logits.

        Args:
            input_ids: Previously generated token ids, shape ``[B, T]``.
                Not used directly but required by the HuggingFace
                ``LogitsProcessor`` interface.
            scores: Raw logits produced by the model, shape ``[B, V]``.

        Returns:
            Modified log-probabilities of shape ``[B, V]``.  Applying
            ``SoftMax`` to the output yields the conditioned distribution
            ``p' = (1 − α) p + α q``.  If no valid green tokens exist the
            original *scores* are returned unchanged.
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
    """Configuration for :class:`UGLD_Against`.

    Attributes:
        red_token_ids: Token ids that form the *red* vocabulary — the set of
            tokens the model is discouraged from generating.  Duplicates and
            out-of-range ids are ignored at runtime.
        lambda_max: Maximum logit penalty λ ≥ 0.  The effective penalty at
            each step is ``lambda_max * φ(p)``, so stronger penalties require
            higher *lambda_max*.  Defaults to ``4.0``.
        tau: Entropy threshold τ for the gate φ.  See
            :class:`UGLDTowardsConfig` for guidance on choosing this value.
            Defaults to ``3.0``.
        s: Smoothing factor s > 0 for the gate sigmoid.  Defaults to ``0.3``.
        eps: Small constant for numerical stability.  Defaults to ``1e-12``.
        weights: How to assign per-token penalty weights within the red
            vocabulary:

            - ``"fixed"`` — every red token receives the same penalty weight
              *fixed_r*.
            - ``"dynamic_minmax"`` — penalty weights are proportional to the
              model's current probability for each red token, with min-max
              normalisation mapping the range to ``[1, 2]``.  Tokens the model
              is most likely to produce receive the heaviest penalty.

            Defaults to ``"fixed"``.
        fixed_r: Penalty weight applied to each red token when
            ``weights="fixed"``.  Must be strictly positive.  Defaults to
            ``1.0``.
    """

    red_token_ids: Sequence[int]
    lambda_max: float = 4.0
    tau: float = 3.0
    s: float = 0.3
    eps: float = 1e-12
    weights: Literal["fixed", "dynamic_minmax"] = "fixed"
    fixed_r: float = 1.0


class UGLD_Against(LogitsProcessor):
    """Condition generation *against* a predefined vocabulary (UGLD-a).

    At each decoding step a penalty is subtracted from the logits of *red*
    tokens.  The penalty strength is gated by the model's predictive
    uncertainty so that suppression is strong when the model is uncertain and
    negligible when it is confident.  Because the penalty is applied in logit
    space, the output remains unnormalised logits and can be passed directly
    to subsequent processors or sampling routines.

    Formally, at each step:

    .. code-block:: text

        p  = SoftMax(z)               # current model distribution
        H  = -Σ p_i log p_i           # Shannon entropy
        φ  = σ((H - τ) / s)           # uncertainty gate ∈ (0, 1)
        λ  = λ_max · φ                # effective penalty strength
        z' = z − λ r                  # penalised logits

    where *r* is a non-negative weight vector supported on the red tokens.

    Args:
        config: A :class:`UGLDAgainstConfig` instance specifying the red
            vocabulary and all hyperparameters.

    Raises:
        ValueError: If ``config.lambda_max < 0`` or ``config.fixed_r <= 0``.

    Example::

        from transformers import LogitsProcessorList
        from ugld import UGLD_Against, UGLDAgainstConfig

        processor = UGLD_Against(UGLDAgainstConfig(
            red_token_ids=red_ids,
            lambda_max=4.0,
            tau=1.0,
            s=0.3,
            weights="fixed",
        ))
        out = model.generate(**inputs, logits_processor=LogitsProcessorList([processor]))
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
        """Apply UGLD-a to a batch of logits.

        Args:
            input_ids: Previously generated token ids, shape ``[B, T]``.
                Not used directly but required by the HuggingFace
                ``LogitsProcessor`` interface.
            scores: Raw logits produced by the model, shape ``[B, V]``.

        Returns:
            Penalised logits of shape ``[B, V]``, equal to
            ``z' = z − λ r``.  If no valid red tokens exist, or if
            ``lambda_max`` is zero, the original *scores* are returned
            unchanged.
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
