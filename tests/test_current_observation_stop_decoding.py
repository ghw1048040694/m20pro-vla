import copy
import pytest
import torch
from m20pro_vla.training.binary_stop import make_binary_stop_spec, VISUAL_LANGUAGE_SCHEMA
from m20pro_vla.training.stop_decoding import CURRENT_STOP_V1,validate_stop_decoding,decoded_stop_logits,save_stop_decoding,load_stop_decoding

def spec():return make_binary_stop_spec(.2,.4,1e-8,schema=VISUAL_LANGUAGE_SCHEMA)

def test_no_sidecar_preserves_exact_original_tensor():
    x=torch.randn(2,50);assert decoded_stop_logits(x,None,spec()) is x

def test_current_prediction_replaces_future_rows_and_can_resume():
    x=torch.ones(2,50);x[0,0]=-2;x[1,0]=2
    y=decoded_stop_logits(x,CURRENT_STOP_V1,spec())
    assert torch.equal(y[0],torch.full((50,),-2.)) and torch.equal(y[1],torch.full((50,),2.))

def test_strict_saved_contract_and_absent_default(tmp_path):
    assert load_stop_decoding(tmp_path,spec()) is None
    save_stop_decoding(tmp_path,CURRENT_STOP_V1,spec())
    assert load_stop_decoding(tmp_path,spec())==CURRENT_STOP_V1

def test_invalid_contract_and_source_head_rejected():
    for change in [{'votes':True},{'chunk_size':10},{'unknown':0},{'permanent_latch':0}]:
        with pytest.raises(ValueError):validate_stop_decoding(dict(CURRENT_STOP_V1,**change),spec())
    with pytest.raises(ValueError):validate_stop_decoding(CURRENT_STOP_V1,make_binary_stop_spec(.2,.4,1e-8))

def test_nonfinite_and_wrong_shape_rejected():
    for x in [torch.zeros(2,10),torch.full((1,50),float('nan'))]:
        with pytest.raises(ValueError):decoded_stop_logits(x,CURRENT_STOP_V1,spec())
