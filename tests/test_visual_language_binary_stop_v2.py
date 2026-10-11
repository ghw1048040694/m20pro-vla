import unittest, torch, copy
from m20pro_vla.training.binary_stop import (make_binary_stop_spec, validate_binary_stop_spec,
    make_binary_stop_head, prefix_stop_logits, VISUAL_LANGUAGE_SCHEMA)
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

class VisualLanguageStopTests(unittest.TestCase):
 def setUp(self):
  self.v1=make_binary_stop_spec(.2,.4,1e-8)
  self.v2=make_binary_stop_spec(.2,.4,1e-8,schema=VISUAL_LANGUAGE_SCHEMA)
  self.head=make_binary_stop_head(self.v2)
  self.hidden=torch.randn(2,8,960,requires_grad=True)
  self.pad=torch.tensor([[1,0,1,0,1,0,1,0],[1,1,0,1,0,1,0,1]],dtype=torch.bool)
  self.att=torch.tensor([[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],dtype=torch.bool)
 def test_versioned_contract_and_cross_version_rejection(self):
  self.assertEqual(validate_binary_stop_spec(self.v2),self.v2)
  self.assertEqual(validate_binary_stop_spec(self.v1),self.v1)
  self.assertTrue(torch.equal(make_binary_stop_head(self.v1).weight,self.head.weight))
  wrong=copy.deepcopy(self.v2);wrong['observation']=self.v1['observation']
  with self.assertRaises(ValueError):validate_binary_stop_spec(wrong)
 def test_group_zero_selection_gaps_gradient_and_state_exclusion(self):
  out=prefix_stop_logits(self.head,self.hidden,self.pad,self.v2,self.att)
  self.assertTrue(torch.equal(out,self.head(torch.stack([self.hidden[0,4],self.hidden[1,5]]))))
  changed=self.hidden.detach().clone();changed[0,6]=1e5;changed[1,7]=-1e5
  self.assertTrue(torch.equal(out.detach(),prefix_stop_logits(self.head,changed,self.pad,self.v2,self.att)))
  out.sum().backward();self.assertEqual(float(self.hidden.grad[0,6].abs().sum()),0)
  self.assertEqual(float(self.hidden.grad[1,7].abs().sum()),0)
  self.assertGreater(float(self.hidden.grad[0,4].abs().sum()),0)
 def test_actual_attention_rule_allows_visual_language_and_blocks_state(self):
  mask=make_att_2d_masks(self.pad,self.att)
  self.assertTrue(torch.equal(mask[0,4],self.pad[0] & (self.att[0].cumsum(0)==0)))
  self.assertTrue(torch.equal(mask[1,5],self.pad[1] & (self.att[1].cumsum(0)==0)))
  self.assertFalse(bool(mask[0,4,6]));self.assertFalse(bool(mask[1,5,7]))
 def test_old_schema_and_implicit_call_preserve_state_token(self):
  expected=self.head(torch.stack([self.hidden[0,6],self.hidden[1,7]]))
  self.assertTrue(torch.equal(expected,prefix_stop_logits(self.head,self.hidden,self.pad)))
  self.assertTrue(torch.equal(expected,prefix_stop_logits(self.head,self.hidden,self.pad,self.v1,self.att)))
 def test_missing_or_ambiguous_groups_are_rejected(self):
  for att in [None,self.att.float(),torch.zeros_like(self.att),torch.ones_like(self.att)]:
   with self.assertRaises(ValueError):prefix_stop_logits(self.head,self.hidden,self.pad,self.v2,att)

if __name__=='__main__':unittest.main()
