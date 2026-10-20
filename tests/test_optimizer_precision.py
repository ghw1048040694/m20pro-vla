import copy
import pytest
import torch
from m20pro_vla.training.optimizer_precision import FP32MasterAdamW,optimizer_precision_spec,validate_precision_spec,load_precision_spec,save_precision_spec

def test_fp32_matches_stock_adamw():
    p=torch.nn.Parameter(torch.tensor([1.,-.3,.0002]));q=torch.nn.Parameter(p.detach().clone())
    a=FP32MasterAdamW([p],lr=1e-5,betas=(.9,.95),weight_decay=1e-10)
    b=torch.optim.AdamW([q],lr=1e-5,betas=(.9,.95),weight_decay=1e-10,foreach=False)
    for i in range(20):
        gradient=torch.tensor([.2+i*.01,-.1,.0003]);p.grad=gradient.clone();q.grad=gradient.clone();a.step();b.step()
        assert torch.equal(p,q)

def test_bfloat16_accumulates_sub_ulp_updates():
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16));q=torch.nn.Parameter(p.detach().float())
    a=FP32MasterAdamW([p],lr=1e-5,betas=(.9,.95));b=torch.optim.AdamW([q],lr=1e-5,betas=(.9,.95),foreach=False)
    legacy=torch.nn.Parameter(p.detach().clone());old=torch.optim.AdamW([legacy],lr=1e-5,betas=(.9,.95),foreach=False)
    for _ in range(220):
        for x in [p,q,legacy]:x.grad=torch.ones_like(x)
        a.step();b.step();old.step()
    assert torch.equal(a.state[p]['fp32_master'],q)
    assert torch.equal(p,q.to(torch.bfloat16)) and not torch.equal(p,legacy)
    assert torch.equal(legacy,torch.ones_like(legacy))
    assert all(a.state[p][k].dtype==torch.float32 for k in ['fp32_master','exp_avg','exp_avg_sq'])

def test_roundtrip_preserves_master_and_resumed_update():
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16));a=FP32MasterAdamW([p],lr=1e-5)
    for _ in range(7):p.grad=torch.ones_like(p);a.step()
    saved=copy.deepcopy(a.state_dict());q=torch.nn.Parameter(p.detach().clone());b=FP32MasterAdamW([q],lr=1e-5);b.load_state_dict(saved)
    assert torch.equal(a.state[p]['fp32_master'],b.state[q]['fp32_master'])
    assert b.state[q]['fp32_master'].dtype==torch.float32
    for _ in range(220):
        p.grad=torch.ones_like(p);q.grad=torch.ones_like(q);a.step();b.step()
    assert torch.equal(p,q) and torch.equal(a.state[p]['fp32_master'],b.state[q]['fp32_master'])

def test_reject_legacy_optimizer_state():
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16));a=torch.optim.AdamW([p]);p.grad=torch.ones_like(p);a.step()
    q=torch.nn.Parameter(p.detach().clone());b=FP32MasterAdamW([q])
    with pytest.raises(ValueError):b.load_state_dict(a.state_dict())

def test_none_grad_and_two_lr_groups():
    p=torch.nn.Parameter(torch.ones(2,dtype=torch.bfloat16));q=torch.nn.Parameter(torch.ones(2))
    a=FP32MasterAdamW([{'params':[p],'lr':1e-5},{'params':[q],'lr':.001}]);q.grad=torch.ones_like(q);a.step()
    assert not a.state[p] and q[0]<.9995
    assert [g['lr'] for g in a.param_groups]==[1e-5,.001]

def test_contract(tmp_path):
    assert load_precision_spec(tmp_path) is None
    save_precision_spec(tmp_path,optimizer_precision_spec());assert load_precision_spec(tmp_path)==optimizer_precision_spec()
    with pytest.raises(ValueError):validate_precision_spec(dict(optimizer_precision_spec(),moments='bfloat16'))
