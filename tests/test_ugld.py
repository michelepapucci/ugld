import pytest
import torch
from ugld import UGLD_Towards, UGLD_Against, UGLDTowardsConfig, UGLDAgainstConfig
from ugld.ugld import _entropy_from_probs, _gate_from_entropy, _valid_token_ids


class TestHelpers:
    def test_entropy_from_probs(self):
        # Test entropy calculation
        p = torch.tensor([[0.1, 0.9], [0.5, 0.5]], dtype=torch.float32)
        eps = 1e-12
        H = _entropy_from_probs(p, eps)
        expected = torch.tensor([
            -(0.1 * torch.log(torch.tensor(0.1)) + 0.9 * torch.log(torch.tensor(0.9))),
            -(0.5 * torch.log(torch.tensor(0.5)) + 0.5 * torch.log(torch.tensor(0.5)))
        ])
        assert torch.allclose(H, expected, atol=1e-6)

    def test_gate_from_entropy(self):
        H = torch.tensor([1.0, 2.0, 3.0])
        tau = 2.0
        s = 0.5
        phi = _gate_from_entropy(H, tau, s)
        expected = torch.sigmoid((H - tau) / s)
        assert torch.allclose(phi, expected)

    def test_gate_s_zero_raises(self):
        with pytest.raises(ValueError, match="s > 0"):
            _gate_from_entropy(torch.tensor([1.0]), 1.0, 0.0)

    def test_valid_token_ids(self):
        ids = [1, 2, 3]
        V = 10
        device = torch.device('cpu')
        result = _valid_token_ids(ids, V, device)
        expected = torch.tensor([1, 2, 3], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_valid_token_ids_clipped(self):
        ids = [1, 2, 15]  # 15 >= 10
        V = 10
        device = torch.device('cpu')
        result = _valid_token_ids(ids, V, device)
        expected = torch.tensor([1, 2], dtype=torch.long)
        assert torch.equal(result, expected)

    def test_valid_token_ids_empty(self):
        ids = []
        V = 10
        device = torch.device('cpu')
        result = _valid_token_ids(ids, V, device)
        assert result.numel() == 0


class TestUGLDTowards:
    @pytest.fixture
    def sample_logits(self):
        B, V = 2, 10
        return torch.randn(B, V)

    @pytest.fixture
    def input_ids(self):
        return torch.zeros(2, 1, dtype=torch.long)

    def test_uniform_prior(self, sample_logits, input_ids):
        config = UGLDTowardsConfig(green_token_ids=[1, 2, 3], prior="uniform")
        processor = UGLD_Towards(config)
        out = processor(input_ids, sample_logits)
        assert out.shape == sample_logits.shape
        assert torch.isfinite(out).all()
        # Check that exp(out) sums to 1 (valid probabilities)
        p_prime = torch.exp(out)
        assert torch.allclose(p_prime.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_renorm_prior(self, sample_logits, input_ids):
        config = UGLDTowardsConfig(green_token_ids=[1, 2, 3], prior="renorm")
        processor = UGLD_Towards(config)
        out = processor(input_ids, sample_logits)
        assert out.shape == sample_logits.shape
        p_prime = torch.exp(out)
        assert torch.allclose(p_prime.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_topk_prior(self, sample_logits, input_ids):
        config = UGLDTowardsConfig(green_token_ids=[1, 2, 3, 4, 5], prior="topk", topk=2)
        processor = UGLD_Towards(config)
        out = processor(input_ids, sample_logits)
        assert out.shape == sample_logits.shape
        p_prime = torch.exp(out)
        assert torch.allclose(p_prime.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_no_green_tokens(self, sample_logits, input_ids):
        config = UGLDTowardsConfig(green_token_ids=[], prior="uniform")
        processor = UGLD_Towards(config)
        out = processor(input_ids, sample_logits)
        # Should return original logits (no-op)
        assert torch.allclose(out, sample_logits, atol=1e-6)

    def test_invalid_prior(self, sample_logits, input_ids):
        config = UGLDTowardsConfig(green_token_ids=[1], prior="invalid")
        processor = UGLD_Towards(config)
        with pytest.raises(ValueError, match="Unknown prior"):
            processor(input_ids, sample_logits)

    def test_alpha_max_validation(self):
        config = UGLDTowardsConfig(green_token_ids=[1], alpha_max=1.5)
        with pytest.raises(ValueError, match="alpha_max must be in"):
            UGLD_Towards(config)
        config2 = UGLDTowardsConfig(green_token_ids=[1], alpha_max=-0.1)
        with pytest.raises(ValueError, match="alpha_max must be in"):
            UGLD_Towards(config2)

    def test_topk_validation(self):
        config = UGLDTowardsConfig(green_token_ids=[1], topk=0)
        with pytest.raises(ValueError, match="topk must be > 0"):
            UGLD_Towards(config)

    def test_batch_processing(self):
        B, V = 3, 5
        logits = torch.randn(B, V)
        input_ids = torch.zeros(B, 1, dtype=torch.long)
        config = UGLDTowardsConfig(green_token_ids=[1, 2], prior="renorm")
        processor = UGLD_Towards(config)
        out = processor(input_ids, logits)
        assert out.shape == (B, V)
        p_prime = torch.exp(out)
        assert torch.allclose(p_prime.sum(dim=-1), torch.ones(B), atol=1e-6)


class TestUGLDAgainst:
    @pytest.fixture
    def sample_logits(self):
        B, V = 2, 10
        return torch.randn(B, V)

    @pytest.fixture
    def input_ids(self):
        return torch.zeros(2, 1, dtype=torch.long)

    def test_fixed_weights(self, sample_logits, input_ids):
        config = UGLDAgainstConfig(red_token_ids=[1, 2, 3], weights="fixed")
        processor = UGLD_Against(config)
        out = processor(input_ids, sample_logits)
        assert out.shape == sample_logits.shape
        assert torch.isfinite(out).all()
        # Softmax should give valid probabilities
        p = torch.softmax(out, dim=-1)
        assert torch.allclose(p.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_dynamic_weights(self, sample_logits, input_ids):
        config = UGLDAgainstConfig(red_token_ids=[1, 2, 3], weights="dynamic_minmax")
        processor = UGLD_Against(config)
        out = processor(input_ids, sample_logits)
        assert out.shape == sample_logits.shape
        p = torch.softmax(out, dim=-1)
        assert torch.allclose(p.sum(dim=-1), torch.ones(2), atol=1e-6)

    def test_no_red_tokens(self, sample_logits, input_ids):
        config = UGLDAgainstConfig(red_token_ids=[], weights="fixed")
        processor = UGLD_Against(config)
        out = processor(input_ids, sample_logits)
        # Should return original logits
        assert torch.allclose(out, sample_logits, atol=1e-6)

    def test_lambda_max_zero(self, sample_logits, input_ids):
        config = UGLDAgainstConfig(red_token_ids=[1], lambda_max=0.0, weights="fixed")
        processor = UGLD_Against(config)
        out = processor(input_ids, sample_logits)
        # No penalty, should return original
        assert torch.allclose(out, sample_logits, atol=1e-6)

    def test_invalid_weights(self, sample_logits, input_ids):
        config = UGLDAgainstConfig(red_token_ids=[1], weights="invalid")
        processor = UGLD_Against(config)
        with pytest.raises(ValueError, match="Unknown weights"):
            processor(input_ids, sample_logits)

    def test_lambda_max_validation(self):
        config = UGLDAgainstConfig(red_token_ids=[1], lambda_max=-0.1)
        with pytest.raises(ValueError, match="lambda_max must be >= 0"):
            UGLD_Against(config)

    def test_fixed_r_validation(self):
        config = UGLDAgainstConfig(red_token_ids=[1], fixed_r=0.0)
        with pytest.raises(ValueError, match="fixed_r must be > 0"):
            UGLD_Against(config)

    def test_batch_processing(self):
        B, V = 3, 5
        logits = torch.randn(B, V)
        input_ids = torch.zeros(B, 1, dtype=torch.long)
        config = UGLDAgainstConfig(red_token_ids=[1, 2], weights="fixed")
        processor = UGLD_Against(config)
        out = processor(input_ids, logits)
        assert out.shape == (B, V)
        p = torch.softmax(out, dim=-1)
        assert torch.allclose(p.sum(dim=-1), torch.ones(B), atol=1e-6)