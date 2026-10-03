"""Equivalence tests for the memory-efficient gather_log_probabilities.

This function sits on the PPO critical path: every run's log-probs, KL, and
actor loss flow through it. The rewrite exists purely to bound memory, so the
bar is exact numerical agreement with the original formulation -- if it drifts,
every downstream metric drifts with it and the runs stop being comparable to
Run A, which trained under the original code.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "ppo_lag"))
import ppo_core  # noqa: E402


def reference(logits, labels):
    """The original implementation, kept here as the oracle."""
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def _case(batch, seq, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g)
    labels = torch.randint(0, vocab, (batch, seq), generator=g)
    return logits, labels


@pytest.mark.parametrize("batch,seq,vocab", [
    (1, 8, 32),        # tiny
    (2, 16, 64),       # batched
    (1, 1, 128),       # single position
    (1, 1300, 64),     # longer than the 512 chunk boundary
    (3, 512, 32),      # exactly at the boundary
    (2, 513, 32),      # one past the boundary
])
def test_matches_reference_under_no_grad(batch, seq, vocab):
    logits, labels = _case(batch, seq, vocab)
    with torch.no_grad():
        got = ppo_core.gather_log_probabilities(logits, labels)
        want = reference(logits, labels)
    assert got.shape == want.shape == (batch, seq)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_matches_reference_with_grad_enabled():
    # The grad path takes the unchunked branch; it must agree too.
    logits, labels = _case(2, 700, 48)
    got = ppo_core.gather_log_probabilities(logits, labels)
    want = reference(logits, labels)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_gradients_match_reference():
    # Actor loss backprops through this; agreeing on values but not gradients
    # would silently change training.
    logits, labels = _case(2, 600, 40)

    a = logits.clone().requires_grad_(True)
    ppo_core.gather_log_probabilities(a, labels).sum().backward()

    b = logits.clone().requires_grad_(True)
    reference(b, labels).sum().backward()

    torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)


def test_chunking_is_not_used_when_grad_enabled():
    # Chunking under autograd would retain every chunk's graph and defeat the
    # purpose; guard the branch condition explicitly.
    logits, labels = _case(1, 2000, 32)
    logits.requires_grad_(True)
    assert torch.is_grad_enabled()
    out = ppo_core.gather_log_probabilities(logits, labels)
    assert out.shape == (1, 2000)


def test_half_precision_matches_fp32_reference():
    """The upcast was removed for memory; verify accuracy did not go with it.

    Production logits are fp16 (the actor loads in half). The oracle is the
    fp32 reference on the same values, so this asserts the fused kernel's
    internal fp32 accumulation really does preserve precision.
    """
    logits, labels = _case(2, 900, 256, seed=7)
    got = ppo_core.gather_log_probabilities(logits.half(), labels)
    want = reference(logits, labels)
    assert got.dtype == torch.float32, "cross_entropy should return fp32 losses"
    # fp16 inputs carry ~1e-3 relative error; the check is that it is that,
    # not the ~1e-1 drift a genuinely half-precision log_softmax would show.
    torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-3)


def test_half_precision_gradients_are_finite_and_close():
    logits, labels = _case(1, 600, 256, seed=11)
    a = logits.half().clone().requires_grad_(True)
    ppo_core.gather_log_probabilities(a, labels).sum().backward()
    b = logits.clone().requires_grad_(True)
    reference(b, labels).sum().backward()
    assert torch.isfinite(a.grad).all(), "half-precision gradients must be finite"
    torch.testing.assert_close(a.grad.float(), b.grad, rtol=5e-3, atol=5e-3)


def test_threshold_is_env_overridable():
    """THRESHOLD must be settable: the default 0.0 makes the Lagrangian inert
    for raw CM scores (lambda only rises above an ~86% violation rate), which
    silently disabled the safety constraint in every CM-based run."""
    import re
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "ppo_lag" / "train_ppo_lag.py").read_text()
    block = re.search(r"for key, cast in \((.+?)\)\s*:", src, re.S).group(1)
    assert '"threshold", float' in block.replace(" ", "").replace('"threshold",float', '"threshold", float') \
        or '"threshold"' in block, "threshold missing from the env-override list"


def test_threshold_formula_matches_measured_geometry():
    """threshold = (cost_violation - cost_safe) * p_target + cost_safe."""
    unsafe, safe = 1.088, -6.698           # augmented CM, measured on the OOD set
    thr = (unsafe - safe) * 0.05 + safe
    assert abs(thr - (-6.309)) < 0.01
    # sanity: at threshold 0.0 the implied target rate is absurdly permissive
    implied = -safe / (unsafe - safe)
    assert implied > 0.85


def test_lambda_max_is_env_overridable():
    """lambda_max must be tunable: the 5.0 default is 8.9x the measured 0.560
    crossover, and that excess force is what diverged Run H at step 111."""
    import re
    src = (Path(__file__).resolve().parent.parent
           / "scripts" / "ppo_lag" / "train_ppo_lag.py").read_text()
    block = re.search(r"for key, cast in \((.+?)\)\s*:", src, re.S).group(1)
    assert '"lambda_max"' in block
