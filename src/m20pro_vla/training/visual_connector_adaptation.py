"""Explicit training of the existing visual connector, with no new model axes."""
import json
from pathlib import Path

CONTRACT_FILE = 'm20_visual_connector_adaptation.json'
PARAMETER = 'model.core.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight'

def visual_connector_spec():
    return dict(schema='m20_visual_connector_adaptation_v1', parameter=PARAMETER,
                shape=[960, 12288], vision_encoder_frozen=True, text_model_frozen=True,
                inference_math='unchanged', learning_rate='existing_body_group')

def validate_connector_spec(spec):
    if not isinstance(spec, dict) or spec != visual_connector_spec():
        raise ValueError('Unknown visual connector adaptation contract')
    # Reject numerically equal noncanonical booleans as well.
    if type(spec['vision_encoder_frozen']) is not bool or type(spec['text_model_frozen']) is not bool:
        raise ValueError('Connector freeze declarations must be booleans')
    return visual_connector_spec()

def load_connector_spec(directory):
    path = Path(directory)/CONTRACT_FILE
    return validate_connector_spec(json.loads(path.read_text())) if path.is_file() else None

def enable_visual_connector_adaptation(policy, spec):
    spec = validate_connector_spec(spec)
    if not policy.config.freeze_vision_encoder or not policy.config.train_expert_only:
        raise ValueError('Connector adaptation requires frozen base vision and text models')
    params = dict(policy.named_parameters())
    if PARAMETER not in params or list(params[PARAMETER].shape) != spec['shape']:
        raise ValueError('The saved model has a different visual connector')
    unexpected = [n for n,p in params.items() if '.vlm.' in n and p.requires_grad and n != PARAMETER]
    if unexpected:
        raise ValueError('Unexpected trainable base VLM parameters: '+str(unexpected))
    params[PARAMETER].requires_grad_(True)
    policy.visual_connector_adaptation = spec
    return params[PARAMETER]

def save_connector_spec(directory, spec):
    spec = validate_connector_spec(spec)
    (Path(directory)/CONTRACT_FILE).write_text(json.dumps(spec,indent=2)+'\n')
