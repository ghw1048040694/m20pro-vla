import unittest
import torch
from m20pro_vla.training.physical_flow_loss import reduce_physical_flow,validate_objective,PHYSICAL_FLOW_V1
class PhysicalLossTests(unittest.TestCase):
 def test_unequal_recorded_duration_keeps_equal_sample_weight(self):
  x=torch.zeros(2,50,32);x[0,:,:4]=2;x[1,:,:4]=6
  pad=torch.zeros(2,50,dtype=torch.bool);pad[1,1:]=True
  self.assertEqual(float(reduce_physical_flow(x,pad)),4)
  self.assertTrue(torch.equal(reduce_physical_flow(x,pad,'none'),torch.tensor([2.,6.])))
 def test_only_recorded_physical_terms_have_gradients(self):
  x=torch.ones(2,50,32,requires_grad=True);pad=torch.zeros(2,50,dtype=torch.bool);pad[1,1:]=True
  reduce_physical_flow(x,pad).backward()
  self.assertEqual(int(torch.count_nonzero(x.grad[:,:,4:])),0)
  self.assertEqual(int(torch.count_nonzero(x.grad[1,1:,:])),0)
  self.assertEqual(int(torch.count_nonzero(x.grad[1,0,:4])),4)
 def test_padding_values_cannot_dilute_or_raise_loss(self):
  x=torch.ones(1,50,32);pad=torch.ones(1,50,dtype=torch.bool);pad[:,0]=False
  x[:,:,4:]=1e8;x[:,1:,:4]=1e9
  self.assertEqual(float(reduce_physical_flow(x,pad)),1)
 def test_missing_mask_or_no_real_action_rejected(self):
  x=torch.ones(1,50,32)
  for mask in [None,torch.ones(1,50,dtype=torch.bool),torch.zeros(1,50)]:
   with self.assertRaises(ValueError):reduce_physical_flow(x,mask)
 def test_unknown_contract_rejected(self):
  self.assertEqual(validate_objective(PHYSICAL_FLOW_V1),PHYSICAL_FLOW_V1)
  with self.assertRaises(ValueError):validate_objective(PHYSICAL_FLOW_V1|{'action_dim':32})
if __name__=='__main__':unittest.main()
