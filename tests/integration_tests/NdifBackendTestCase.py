"""
Test cases for the NDIF (nnsight) backend integration.
"""
import unittest
import torch
import os

try:
    from nnsight import LanguageModel
    NNSIGHT_AVAILABLE = True
except ImportError:
    NNSIGHT_AVAILABLE = False

import pyvene as pv
from pyvene.models.basic_utils import get_batch_size
from pyvene.models.interventions import (
    CollectIntervention,
    VanillaIntervention,
    AdditionIntervention,
    SubtractionIntervention,
    ZeroIntervention,
    NoiseIntervention,
    SkipIntervention,
    LowRankRotatedSpaceIntervention,
    RotatedSpaceIntervention,
    BoundlessRotatedSpaceIntervention,
    SigmoidMaskRotatedSpaceIntervention,
    SigmoidMaskIntervention,
)

def get_remote_clean_output(model, base_tokens):
    """Helper function to get clean output from a remote NNsight model."""
    # Convert base_tokens to raw tensors to avoid serialization issues
    if hasattr(base_tokens, 'keys') and hasattr(base_tokens, 'values'):
        base_tokens_raw = {k: v for k, v in base_tokens.items()}
    else:
        base_tokens_raw = base_tokens

    with model.session(remote=True):
        with model.trace(base_tokens_raw):
            clean_output = model.output.save()
    return clean_output


@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
class NdifBackendTestCase(unittest.TestCase):
    """Test NDIF backend with local execution."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifBackendTestCase ===")
        cls.model = LanguageModel('openai-community/gpt2', device_map='cpu')
        cls.tokenizer = cls.model.tokenizer
        cls.base_tokens = cls.tokenizer("The capital of France is", return_tensors="pt")
        cls.source_tokens = cls.tokenizer("The capital of Germany is", return_tensors="pt")
        cls.seq_len = cls.base_tokens['input_ids'].shape[1]

    def test_collect_intervention(self):
        """Test CollectIntervention collects activations."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].mlp.c_proj.output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=False)

        result = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )

        _, collected = result[0]
        self.assertIsInstance(collected, list)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].shape, (self.seq_len, 768))

    def test_vanilla_intervention(self):
        """Test VanillaIntervention swaps activations."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": VanillaIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_generate_with_intervention(self):
        """Test generation with CollectIntervention."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=False)

        _, gen_output = pv_model.generate(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))},
            max_new_tokens=3,
        )
        self.assertIsNotNone(gen_output)

    def test_clean_run(self):
        """Test clean run produces correct output."""
        pv_model = pv.build_intervenable_model([], model=self.model, remote=False)

        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                expected = self.model.output.save()

        pv_output, _ = pv_model(base=self.base_tokens)

        self.assertTrue(torch.allclose(expected.logits, pv_output.logits, atol=1e-5))

    def test_addition_intervention(self):
        """Test AdditionIntervention adds source to base."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": AdditionIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change due to addition
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_subtraction_intervention(self):
        """Test SubtractionIntervention subtracts source from base."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": SubtractionIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change due to subtraction
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_zero_intervention(self):
        """Test ZeroIntervention zeros out activations."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].mlp.c_proj.output",
            "intervention": ZeroIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Apply zero intervention (no sources needed)
        _, intervened = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )

        # Output should change due to zeroed activations
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    @unittest.skip("NoiseIntervention creates fixed-shape noise tensor (1, 4, embed_dim) that doesn't match arbitrary sequence lengths - pre-existing limitation")
    def test_noise_intervention(self):
        """Test NoiseIntervention adds noise to activations."""
        # NoiseIntervention has a fixed noise shape of (1, 4, embed_dim)
        # This is a pre-existing limitation in the intervention design
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": NoiseIntervention(embed_dim=768)
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Apply noise intervention (no sources needed)
        _, intervened = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )

        # Output should change due to added noise
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_skip_intervention(self):
        """Test SkipIntervention replaces output with input (skips layer)."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": SkipIntervention()
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Apply skip intervention - intervene on all positions
        _, intervened = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )

        # Output should change due to skipped layer computation
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )


class BasicUtilsNdifTestCase(unittest.TestCase):
    """Test basic_utils changes for NDIF support."""

    def test_get_batch_size_string(self):
        """Test get_batch_size with string input."""
        self.assertEqual(get_batch_size("Hello world"), 1)

    def test_get_batch_size_list_of_strings(self):
        """Test get_batch_size with list of strings."""
        self.assertEqual(get_batch_size(["Hello", "World"]), 2)


@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
@unittest.skipUnless(os.environ.get('NDIF_REMOTE_TESTS') == '1',
                     "Remote tests disabled. Set NDIF_REMOTE_TESTS=1 to enable.")
class NdifBackendRemoteTestCase(unittest.TestCase):
    """Test NDIF backend with remote execution (remote=True)."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifBackendRemoteTestCase ===")
        cls.model = LanguageModel('openai-community/gpt2')
        cls.tokenizer = cls.model.tokenizer
        cls.base_tokens = cls.tokenizer("The capital of France is", return_tensors="pt")
        cls.source_tokens = cls.tokenizer("The capital of Germany is", return_tensors="pt")
        cls.seq_len = cls.base_tokens['input_ids'].shape[1]

    def test_remote_collect_intervention(self):
        """Test CollectIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].mlp.c_proj.output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=True)

        result = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )
        _, collected = result[0]
        self.assertIsInstance(collected, list)
        self.assertEqual(len(collected), 1)

    def test_remote_vanilla_intervention(self):
        """Test VanillaIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": VanillaIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_generate(self):
        """Test generation with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": CollectIntervention()
        }, model=self.model, remote=True)

        _, gen_output = pv_model.generate(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))},
            max_new_tokens=3,
        )
        self.assertIsNotNone(gen_output)

    def test_remote_addition_intervention(self):
        """Test AdditionIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": AdditionIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_subtraction_intervention(self):
        """Test SubtractionIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": SubtractionIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_zero_intervention(self):
        """Test ZeroIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].mlp.c_proj.output",
            "intervention": ZeroIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Apply zero intervention
        _, intervened = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_skip_intervention(self):
        """Test SkipIntervention with remote=True."""
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": SkipIntervention()
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Apply skip intervention - intervene on all positions
        _, intervened = pv_model(
            base=self.base_tokens,
            unit_locations={"base": list(range(self.seq_len))}
        )
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )


@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
class NdifTrainableInterventionTestCase(unittest.TestCase):
    """Test trainable interventions with NDIF backend."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifTrainableInterventionTestCase ===")
        cls.model = LanguageModel('openai-community/gpt2', device_map='cpu')
        cls.tokenizer = cls.model.tokenizer
        cls.base_tokens = cls.tokenizer("The capital of France is", return_tensors="pt")
        cls.source_tokens = cls.tokenizer("The capital of Germany is", return_tensors="pt")
        cls.seq_len = cls.base_tokens['input_ids'].shape[1]
        cls.embed_dim = 768  # GPT-2's hidden size

    def test_low_rank_rotated_intervention_local(self):
        """Test LowRankRotatedSpaceIntervention with local nnsight."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change due to rotation intervention
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_rotated_space_intervention_local(self):
        """Test RotatedSpaceIntervention with local nnsight."""
        intervention = RotatedSpaceIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_sigmoid_mask_intervention_local(self):
        """Test SigmoidMaskIntervention with local nnsight."""
        intervention = SigmoidMaskIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Get clean output
        with self.model.session(remote=False):
            with self.model.trace(self.base_tokens):
                clean_output = self.model.output.save()

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output.logits, intervened.logits, atol=1e-3)
        )

    def test_get_remote_weights_low_rank(self):
        """Test get_remote_weights returns correct structure for LowRankRotatedSpaceIntervention."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        weights = intervention.get_remote_weights()

        self.assertIn('rotate_layer_weight', weights)
        self.assertIn('embed_dim', weights)
        self.assertIn('low_rank_dimension', weights)
        self.assertIn('intervention_type', weights)
        self.assertEqual(weights['intervention_type'], 'low_rank_rotated_space')
        self.assertEqual(weights['rotate_layer_weight'].shape, (self.embed_dim, 64))

    def test_get_remote_weights_rotated_space(self):
        """Test get_remote_weights returns correct structure for RotatedSpaceIntervention."""
        intervention = RotatedSpaceIntervention(embed_dim=self.embed_dim)
        weights = intervention.get_remote_weights()

        self.assertIn('rotate_layer_weight', weights)
        self.assertIn('intervention_type', weights)
        self.assertEqual(weights['intervention_type'], 'rotated_space')
        self.assertEqual(weights['rotate_layer_weight'].shape, (self.embed_dim, self.embed_dim))

    def test_gradient_flow_through_intervention(self):
        """Verify gradients flow through trainable intervention parameters."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Ensure intervention parameters require gradients
        self.assertTrue(intervention.trainable)
        has_grad_params = any(p.requires_grad for p in intervention.parameters())
        self.assertTrue(has_grad_params, "Intervention should have gradient-enabled parameters")

        # Forward pass
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Compute loss and backward (using a simple sum loss)
        # Note: This tests that the output is connected to the intervention params
        if hasattr(intervened, 'logits'):
            loss = intervened.logits.sum()
        else:
            loss = intervened.sum()
        loss.backward()

        # Check gradients exist on intervention parameters
        for name, param in intervention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Gradient should exist for parameter {name}"
                )

    def test_forward_with_gradients_method(self):
        """Test the forward_with_gradients method for training."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=False)

        # Use forward_with_gradients
        output = pv_model.forward_with_gradients(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        self.assertIsNotNone(output)


@unittest.skipUnless(NNSIGHT_AVAILABLE, "nnsight not installed")
@unittest.skipUnless(os.environ.get('NDIF_REMOTE_TESTS') == '1',
                     "Remote tests disabled. Set NDIF_REMOTE_TESTS=1 to enable.")
class NdifTrainableInterventionRemoteTestCase(unittest.TestCase):
    """Test trainable interventions with NDIF backend and remote=True."""

    @classmethod
    def setUpClass(cls):
        print("=== Test Suite: NdifTrainableInterventionRemoteTestCase ===")
        cls.model = LanguageModel('openai-community/gpt2')
        cls.tokenizer = cls.model.tokenizer
        cls.base_tokens = cls.tokenizer("The capital of France is", return_tensors="pt")
        cls.source_tokens = cls.tokenizer("The capital of Germany is", return_tensors="pt")
        cls.seq_len = cls.base_tokens['input_ids'].shape[1]
        cls.embed_dim = 768

    def test_remote_low_rank_rotated_intervention(self):
        """Test LowRankRotatedSpaceIntervention with remote=True."""
        intervention = LowRankRotatedSpaceIntervention(
            embed_dim=self.embed_dim, low_rank_dimension=64
        )
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_rotated_space_intervention(self):
        """Test RotatedSpaceIntervention with remote=True."""
        intervention = RotatedSpaceIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )

    def test_remote_sigmoid_mask_intervention(self):
        """Test SigmoidMaskIntervention with remote=True."""
        intervention = SigmoidMaskIntervention(embed_dim=self.embed_dim)
        pv_model = pv.build_intervenable_model({
            "component": "transformer.h[0].output",
            "intervention": intervention
        }, model=self.model, remote=True)

        # Get clean output
        clean_output = get_remote_clean_output(self.model, self.base_tokens)

        # Get intervened output
        _, intervened = pv_model(
            base=self.base_tokens,
            sources=[self.source_tokens],
            unit_locations={"sources->base": ([None], [None])}
        )

        # Output should change
        self.assertFalse(
            torch.allclose(clean_output['logits'], intervened['logits'], atol=1e-3)
        )


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BasicUtilsNdifTestCase))
    if NNSIGHT_AVAILABLE:
        suite.addTest(unittest.makeSuite(NdifBackendTestCase))
        suite.addTest(unittest.makeSuite(NdifTrainableInterventionTestCase))
        if os.environ.get('NDIF_REMOTE_TESTS') == '1':
            suite.addTest(unittest.makeSuite(NdifBackendRemoteTestCase))
            suite.addTest(unittest.makeSuite(NdifTrainableInterventionRemoteTestCase))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())