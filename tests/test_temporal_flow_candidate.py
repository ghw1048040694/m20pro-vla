import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from m20pro_vla.training.temporal_flow_candidate import TemporalFlowCoreCandidate


class InstrumentedCore(nn.Module):
    """Tests the bridge contract, not a vision/flow model substitute."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
    def forward(self, images, masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        return self.embed_prefix(images, masks, lang_tokens, lang_masks, state=state)
    def sample_actions(self, images, masks, lang_tokens, lang_masks, state, noise=None, **kwargs):
        return self.embed_prefix(images, masks, lang_tokens, lang_masks, state=state)


class TemporalFlowCandidateTests(unittest.TestCase):
    def candidate(self): return TemporalFlowCoreCandidate(InstrumentedCore(), 6)
    def inputs(self):
        return [torch.zeros((1,3,2,2)) for _ in range(6)], [torch.tensor([True]) for _ in range(6)]

    def test_train_and_sampling_build_identical_age_context(self):
        candidate=self.candidate();images,masks=self.inputs()
        masks[:2]=[torch.tensor([False]),torch.tensor([False])]
        original=candidate.core.forward.__func__
        def capture(core,identity,images,masks,ages,*args):return ages.clone()
        with patch('m20pro_vla.training.temporal_flow_candidate.embed_prefix_with_identity',side_effect=capture):
            train=candidate(images,masks,None,None,torch.zeros((1,32)),torch.zeros((1,2,4)))
            inference=candidate.sample_actions(images,masks,None,None,torch.zeros((1,32)))
        self.assertTrue(torch.equal(train,inference))
        self.assertTrue(torch.equal(train,torch.tensor([[-1,-1,100,100,0,0]])))
        self.assertIs(candidate.core.forward.__func__,original)

    def test_parameters_include_new_identity_and_base_without_duplicates(self):
        candidate=self.candidate();params=list(candidate.parameters())
        self.assertEqual(len(params),len(list(candidate.identity.parameters()))+1)
        self.assertEqual(len(params),len(set(map(id,params))))
        self.assertIn('identity.age_projection.weight',candidate.state_dict())
        self.assertIn('core.weight',candidate.state_dict())

    def test_contract_reconstruction_then_state_load_is_exact(self):
        candidate=self.candidate();restored=TemporalFlowCoreCandidate.from_contract(InstrumentedCore(),candidate.contract())
        restored.load_state_dict(candidate.state_dict())
        self.assertEqual(candidate.contract(),restored.contract())
        self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in candidate.state_dict().items()))

    def test_missing_current_mismatched_camera_or_incomplete_images_rejected(self):
        candidate=self.candidate();images,masks=self.inputs()
        with self.assertRaises(ValueError):candidate._view(images[:-1],masks[:-1])
        masks[0]=torch.tensor([False])
        with self.assertRaises(ValueError):candidate._view(images,masks)
        masks[1]=torch.tensor([False]);masks[-1]=torch.tensor([False])
        with self.assertRaises(ValueError):candidate._view(images,masks)


if __name__=='__main__':unittest.main()
