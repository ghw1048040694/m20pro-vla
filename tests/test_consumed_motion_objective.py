import copy
import pytest
import torch
from m20pro_vla.training.physical_flow_loss import (
    PHYSICAL_FLOW_V1, CONSUMED_MOTION_V2, validate_objective,
    reduce_physical_flow, validate_objective_context)

def test_consumed_gradient_keeps_v1_scale_and_masks_discarded_axes():
    x=torch.arange(2*50*32,dtype=torch.float64).reshape(2,50,32).requires_grad_()
    pad=torch.ones(2,50,dtype=torch.bool);pad[0,:3]=False;pad[1,:1]=False
    new=reduce_physical_flow(x,pad,objective=CONSUMED_MOTION_V2)
    old=reduce_physical_flow(x,pad,objective=PHYSICAL_FLOW_V1)
    ng=torch.autograd.grad(new,x,retain_graph=True)[0];og=torch.autograd.grad(old,x)[0]
    assert torch.equal(ng[:,:,[0,2]],og[:,:,[0,2]])
    assert torch.count_nonzero(ng[:,:,[1,3,*range(4,32)]])==0
    assert torch.count_nonzero(ng[pad])==0
    per=reduce_physical_flow(x,pad,'none',objective=CONSUMED_MOTION_V2)
    expected=torch.stack([x[0,:3][:,[0,2]].sum()/12,x[1,:1][:,[0,2]].sum()/4])
    assert torch.equal(per,expected) and torch.equal(new,expected.mean())

def test_unused_and_padded_extremes_do_not_pollute_consumed_loss():
    x=torch.ones(2,50,32,requires_grad=True)
    pad=torch.ones(2,50,dtype=torch.bool);pad[:,0]=False
    with torch.no_grad():
        x[:,:,1]=float('nan');x[:,:,3]=float('inf');x[:,1:,0]=float('nan')
    loss=reduce_physical_flow(x,pad,objective=CONSUMED_MOTION_V2)
    assert loss.item()==.5
    loss.backward();assert torch.isfinite(x.grad).all()

def test_strict_versions_and_empty_samples():
    assert validate_objective(CONSUMED_MOTION_V2)==CONSUMED_MOTION_V2
    for key,value in [('consumed_axes',[0,1]),('denominator_axes',2),('extra',0)]:
        bad=copy.deepcopy(CONSUMED_MOTION_V2);bad[key]=value
        with pytest.raises(ValueError):validate_objective(bad)
    with pytest.raises(ValueError):
        reduce_physical_flow(torch.zeros(1,50,32),torch.ones(1,50,dtype=torch.bool),objective=CONSUMED_MOTION_V2)

def test_context_requires_head_and_lateral_zero(monkeypatch):
    from m20pro_vla.training import binary_stop
    spec=binary_stop.make_binary_stop_spec(.2,.4,1e-8,schema=binary_stop.VISUAL_LANGUAGE_SCHEMA)
    monkeypatch.setattr(binary_stop,'_saved_action_stats',lambda *a:([torch.zeros(4),torch.zeros(4)],1e-8))
    validate_objective_context('.',CONSUMED_MOTION_V2,spec)
    with pytest.raises(ValueError):validate_objective_context('.',CONSUMED_MOTION_V2,None)
    wrong=binary_stop.make_binary_stop_spec(.2,.4,1e-8)
    with pytest.raises(ValueError):validate_objective_context('.',CONSUMED_MOTION_V2,wrong)
    monkeypatch.setattr(binary_stop,'_saved_action_stats',lambda *a:([torch.zeros(4),torch.ones(4)],1e-8))
    with pytest.raises(ValueError):validate_objective_context('.',CONSUMED_MOTION_V2,spec)
    validate_objective_context('.',PHYSICAL_FLOW_V1,None)
