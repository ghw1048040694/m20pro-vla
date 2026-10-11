import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from m20pro_vla.training.visual_connector_adaptation import (
    PARAMETER, visual_connector_spec, validate_connector_spec,
    enable_visual_connector_adaptation, save_connector_spec, load_connector_spec)

def test_strict_spec():
    spec=visual_connector_spec();assert validate_connector_spec(spec)==spec
    for bad in [dict(spec,shape=[960,3072]),dict(spec,inference_math='mask_state'),
                dict(spec,vision_encoder_frozen=1),dict(spec,extra=True)]:
        with pytest.raises(ValueError):validate_connector_spec(bad)

def test_strict_sidecar(tmp_path):
    assert load_connector_spec(tmp_path) is None
    save_connector_spec(tmp_path,visual_connector_spec())
    assert load_connector_spec(tmp_path)==visual_connector_spec()

def test_enable_exact_parameter_only():
    connector=torch.nn.Parameter(torch.zeros(960,12288),requires_grad=False)
    other=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
    policy=SimpleNamespace(config=SimpleNamespace(freeze_vision_encoder=True,train_expert_only=True),
        named_parameters=lambda:iter([(PARAMETER,connector),('model.core.vlm_with_expert.vlm.model.text.other',other)]))
    assert enable_visual_connector_adaptation(policy,visual_connector_spec()) is connector
    assert connector.requires_grad and not other.requires_grad
    assert policy.visual_connector_adaptation==visual_connector_spec()

def test_reject_other_unfrozen_vlm():
    p=torch.nn.Parameter(torch.zeros(960,12288),requires_grad=False)
    other=torch.nn.Parameter(torch.zeros(1),requires_grad=True)
    policy=SimpleNamespace(config=SimpleNamespace(freeze_vision_encoder=True,train_expert_only=True),
        named_parameters=lambda:iter([(PARAMETER,p),('model.core.vlm_with_expert.vlm.model.text.other',other)]))
    with pytest.raises(ValueError):enable_visual_connector_adaptation(policy,visual_connector_spec())

def test_reject_wrong_base_flags_or_shape():
    p=torch.nn.Parameter(torch.zeros(960,1),requires_grad=False)
    policy=SimpleNamespace(config=SimpleNamespace(freeze_vision_encoder=True,train_expert_only=True),named_parameters=lambda:iter([(PARAMETER,p)]))
    with pytest.raises(ValueError):enable_visual_connector_adaptation(policy,visual_connector_spec())
    policy.config.train_expert_only=False
    with pytest.raises(ValueError):enable_visual_connector_adaptation(policy,visual_connector_spec())
