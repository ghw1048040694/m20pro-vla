import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from m20pro_vla.training.binary_stop import (
    CONTRACT_FILE, binary_stop_loss, load_binary_stop_spec, make_binary_stop_head,
    make_binary_stop_spec, prefix_stop_logits, replace_stop_column,
    save_binary_stop_spec, validate_binary_stop_spec, make_binary_stop_spec_from_saved,
    validate_saved_normalization,
)


class BinaryStopTests(unittest.TestCase):
    def setUp(self):
        self.spec = make_binary_stop_spec(.2, .4, 1e-8)

    def actions(self, labels):
        result = torch.zeros(*labels.shape, 4)
        result[..., 3] = self.spec['normalized_stop_zero'] + labels * (
            self.spec['normalized_stop_one'] - self.spec['normalized_stop_zero'])
        return result

    def test_saved_contract_is_strict_and_validates_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_binary_stop_spec(directory))
            save_binary_stop_spec(directory, self.spec)
            self.assertEqual(load_binary_stop_spec(directory), self.spec)
        for key, value in [('hidden_dim', 32), ('loss_coefficient', 2.),
                           ('normalized_stop_zero', .5), ('normalization_epsilon', float('nan'))]:
            wrong = copy.deepcopy(self.spec)
            wrong[key] = value
            with self.assertRaises(ValueError):
                validate_binary_stop_spec(wrong)
        with self.assertRaises(ValueError):
            validate_binary_stop_spec(dict(self.spec, unknown=True))

    def test_initialization_is_deterministic_without_consuming_rng(self):
        before = torch.get_rng_state().clone()
        head = make_binary_stop_head(self.spec)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        other = make_binary_stop_head(self.spec)
        self.assertTrue(torch.equal(head.weight, other.weight))
        self.assertEqual(int(torch.count_nonzero(head.bias)), 0)
        self.assertGreater(float(head.weight.std()), .0009)
        self.assertLess(float(head.weight.std()), .0011)

    def test_last_true_prefix_position_with_interior_gaps_and_graph(self):
        head = make_binary_stop_head(self.spec)
        hidden = torch.randn(2, 8, 960, requires_grad=True)
        mask = torch.tensor([[1, 0, 0, 1, 0, 1, 0, 0],
                             [0, 1, 0, 0, 1, 0, 0, 1]], dtype=torch.bool)
        logits = prefix_stop_logits(head, hidden, mask)
        self.assertTrue(torch.equal(logits, head(torch.stack([hidden[0, 5], hidden[1, 7]]))))
        changed = hidden.detach().clone()
        changed[~mask] = 1e8
        self.assertTrue(torch.equal(logits.detach(), prefix_stop_logits(head, changed, mask)))
        logits.sum().backward()
        self.assertGreater(float(hidden.grad[0, 5].abs().sum()), 0)
        self.assertGreater(float(hidden.grad[1, 7].abs().sum()), 0)
        hidden.grad[0, 5] = 0
        hidden.grad[1, 7] = 0
        self.assertEqual(int(torch.count_nonzero(hidden.grad)), 0)
        with self.assertRaises(ValueError):
            prefix_stop_logits(head, hidden, torch.zeros_like(mask))

    def test_binary_labels_use_saved_normalization_not_point_five(self):
        labels = torch.arange(50).remainder(2)[None].float()
        action = self.actions(labels)
        logits = (labels * 2 - 1) * 2
        mask = torch.zeros(1, 50, dtype=torch.bool)
        expected = F.binary_cross_entropy_with_logits(logits, labels)
        self.assertTrue(torch.equal(binary_stop_loss(logits, action, mask, self.spec), expected))
        action[0, 3, 3] = (self.spec['normalized_stop_zero'] + self.spec['normalized_stop_one']) / 2
        with self.assertRaises(ValueError):
            binary_stop_loss(logits, action, mask, self.spec)

    def test_padding_extremes_do_not_affect_loss_or_gradient(self):
        logits = torch.zeros(1, 50, requires_grad=True)
        pad = torch.ones(1, 50, dtype=torch.bool)
        pad[:, :2] = False
        actions = self.actions(torch.ones(1, 50))
        reference = binary_stop_loss(logits, actions, pad, self.spec)
        changed = logits.detach().clone()
        changed[:, 2:] = 1e20
        changed.requires_grad_()
        actions[:, 2:, 3] = float('nan')
        loss = binary_stop_loss(changed, actions, pad, self.spec)
        self.assertTrue(torch.equal(reference, loss))
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(changed.grad[:, 2:])), 0)
        self.assertEqual(int(torch.count_nonzero(changed.grad[:, :2])), 2)

    def test_equal_sample_weight_with_unequal_recorded_lengths(self):
        logits = torch.zeros(2, 50)
        logits[1] = -2
        labels = torch.ones(2, 50)
        pad = torch.zeros(2, 50, dtype=torch.bool)
        pad[1, 1:] = True
        losses = binary_stop_loss(logits, self.actions(labels), pad, self.spec, reduction='none')
        expected = F.binary_cross_entropy_with_logits(torch.tensor([0., -2.]), torch.ones(2), reduction='none')
        torch.testing.assert_close(losses, expected)
        torch.testing.assert_close(binary_stop_loss(logits, self.actions(labels), pad, self.spec), expected.mean())
        for mask in [None, torch.ones_like(pad), pad.float()]:
            with self.assertRaises(ValueError):
                binary_stop_loss(logits, self.actions(labels), mask, self.spec)

    def test_replacement_changes_only_stop_and_roundtrips_output_probability(self):
        # A deliberately material epsilon exposes forward/inverse asymmetry.
        spec = make_binary_stop_spec(.2, .4, .01)
        actions = torch.randn(2, 50, 32, requires_grad=True)
        original = actions.detach().clone()
        logits = torch.linspace(-10, 10, 100).reshape(2, 50).requires_grad_()
        replaced = replace_stop_column(actions, logits, spec)
        columns = torch.arange(32) != 3
        self.assertTrue(torch.equal(replaced[..., columns], actions[..., columns]))
        self.assertTrue(torch.equal(actions.detach(), original))
        physical_stop = replaced[..., 3] * spec['stop_std'] + spec['stop_mean']
        torch.testing.assert_close(physical_stop, logits.sigmoid(), atol=1e-7, rtol=1e-6)
        replaced.sum().backward()
        self.assertEqual(int(torch.count_nonzero(actions.grad[..., 3])), 0)
        self.assertGreater(float(logits.grad.abs().sum()), 0)

    def test_saved_processor_stats_and_endpoints_must_agree(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = {'action.mean': torch.tensor([.1, 0., .2, .2]),
                     'action.std': torch.tensor([.3, 0., .4, .4])}
            norm = {'eps': 1e-8, 'norm_map': {'ACTION': 'MEAN_STD'},
                    'features': {'action': {'type': 'ACTION', 'shape': [4]}}}
            (root / 'config.json').write_text(json.dumps({'max_action_dim': 32, 'chunk_size': 50,
                'output_features': {'action': {'type': 'ACTION', 'shape': [4]}}}))
            for name, registry in [('preprocessor', 'normalizer_processor'),
                                   ('postprocessor', 'unnormalizer_processor')]:
                tensor_name = name + '.safetensors'
                save_file(stats, root / tensor_name)
                (root / ('policy_' + name + '.json')).write_text(json.dumps({'steps': [{
                    'registry_name': registry, 'config': norm, 'state_file': tensor_name}]}))
            spec = make_binary_stop_spec_from_saved(root)
            self.assertEqual(spec, self.spec)
            self.assertEqual(validate_saved_normalization(root, spec), spec)
            save_binary_stop_spec(root, spec)
            self.assertTrue((root / CONTRACT_FILE).is_file())
            stats['action.mean'][0] = .9
            save_file(stats, root / 'postprocessor.safetensors')
            with self.assertRaises(ValueError):
                make_binary_stop_spec_from_saved(root)


if __name__ == '__main__':
    unittest.main()
