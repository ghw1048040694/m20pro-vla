"""Opt-in FP32 AdamW accumulation with original low-precision forward weights."""
import json
from pathlib import Path
from dataclasses import asdict
import torch

CONTRACT_FILE = 'm20_optimizer_precision.json'
def optimizer_precision_spec():
    return dict(schema='m20_fp32_master_adamw_v1', optimizer='adamw',
                forward_weights='original_dtype', master_weights='float32',
                moments='float32', rounded_copy='after_each_optimizer_step')
def validate_precision_spec(spec):
    if not isinstance(spec,dict) or spec != optimizer_precision_spec():
        raise ValueError('Unknown optimizer precision contract')
    return optimizer_precision_spec()
def load_precision_spec(directory):
    p=Path(directory)/CONTRACT_FILE
    return validate_precision_spec(json.loads(p.read_text())) if p.is_file() else None
def save_precision_spec(directory,spec):
    (Path(directory)/CONTRACT_FILE).write_text(json.dumps(validate_precision_spec(spec),indent=2)+'\n')

class FP32MasterAdamW(torch.optim.AdamW):
    """Use stock AdamW on FP32 storage, then round only the forward copy.

    Parameter objects and optimizer group ordering remain original. The FP32
    master and moments are serialized in the ordinary optimizer state dict.
    This implementation intentionally excludes closures, sparse gradients,
    CUDA capture, fused/foreach and differentiable optimizer steps.
    """
    def __init__(self,params,**kwargs):
        if any(kwargs.get(k,False) for k in ['capturable','differentiable','fused','foreach']):
            raise ValueError('Master AdamW requires ordinary single-tensor AdamW')
        super().__init__(params,**(kwargs|dict(foreach=False,capturable=False,differentiable=False,fused=False)))

    @torch.no_grad()
    def step(self,closure=None):
        if closure is not None:raise ValueError('Master AdamW does not support closures')
        swapped=[]
        try:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:continue
                    if p.grad.is_sparse:raise ValueError('Sparse master gradients are unsupported')
                    if p.dtype not in [torch.float32,torch.bfloat16,torch.float16]:
                        raise ValueError('Unsupported master parameter dtype')
                    if p.dtype==torch.float32:continue
                    state=self.state[p]
                    master=state.get('fp32_master')
                    if master is None:
                        if state:raise ValueError('Cannot silently upgrade an existing low-precision optimizer state')
                        master=p.detach().float().clone()
                    if master.dtype!=torch.float32:raise ValueError('Master weight lost FP32 precision')
                    forward,gradient=p.data,p.grad
                    p.data=master
                    p.grad=gradient.float()
                    swapped.append((p,forward,gradient))
            result=super().step()
        finally:
            for p,forward,gradient in swapped:
                # No alias between the serialized master and forward storage.
                self.state[p]['fp32_master']=p.data
                forward.copy_(p.data)
                p.data=forward
                p.grad=gradient
        return result

    def load_state_dict(self,state_dict):
        # Optimizer.load_state_dict normally casts every floating state tensor
        # to the forward parameter dtype. Restore saved FP32 tensors directly.
        incoming=state_dict['state']
        mapping=[]
        for saved,current in zip(state_dict['param_groups'],self.param_groups,strict=True):
            if len(saved['params'])!=len(current['params']):raise ValueError('Optimizer groups differ')
            mapping.extend(zip(saved['params'],current['params'],strict=True))
        for key,p in mapping:
            s=incoming.get(key,{})
            if not s:continue
            required={'step','exp_avg','exp_avg_sq'}|({'fp32_master'} if p.dtype!=torch.float32 else set())
            if set(s)!=required:raise ValueError('Incomplete or incompatible master optimizer state')
            for name in required-{'step'}:
                if s[name].dtype!=torch.float32 or s[name].shape!=p.shape:
                    raise ValueError('Saved master/moment tensor is not exact FP32')
        super().load_state_dict(state_dict)
        for key,p in mapping:
            for name,t in incoming.get(key,{}).items():
                if name!='step':self.state[p][name]=t.to(device=p.device,dtype=torch.float32).clone()
        return None

def make_precision_optimizer_and_scheduler(cfg,policy):
    validate_precision_spec(policy.optimizer_precision)
    if cfg.optimizer.type!='adamw' or cfg.policy.use_amp or cfg.policy.use_peft:
        raise ValueError('Master precision currently requires non-AMP non-PEFT AdamW')
    params=policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    kwargs=asdict(cfg.optimizer);kwargs.pop('grad_clip_norm')
    optimizer=FP32MasterAdamW(params,**kwargs)
    scheduler=cfg.scheduler.build(optimizer,cfg.steps) if cfg.scheduler is not None else None
    return optimizer,scheduler
