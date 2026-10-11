"""Integrity review for explicitly selected human adaptation data."""
import hashlib
import json
from pathlib import Path
from .execution_contract import HUMAN_CONTRACT, load_execution_contract


def reviewed_human_source(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['schema']!='m20_reviewed_human_raw_v1' or manifest['heldout_included']:
        raise ValueError('Expected reviewed human training data, excluding heldout')
    if load_execution_contract(directory)!=HUMAN_CONTRACT:
        raise ValueError('Human source has no matching physical execution contract')
    records=manifest['records']
    if not records or len({r['episode_id'] for r in records})!=len(records):
        raise ValueError('Empty or duplicated human training source')
    for r in records:
        for suffix,key in [('.npz','npz_sha256'),('.json','metadata_sha256')]:
            path=directory/f"episode_{r['episode_id']}{suffix}"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=r[key]:
                raise ValueError('Reviewed human file changed: '+str(path))
        m=json.loads((directory/f"episode_{r['episode_id']}.json").read_text())
        if (m['collection_mode']!='human_keyboard_reviewed' or not m['training_ready']
                or not 6012<=int(m['layout_id'])<=6023
                or m['source_recording_id']!=r['source_recording_id']):
            raise ValueError('Human source provenance differs')
        review=m['human_review']
        if review['actual_contact_steps']!=0 or review['terminal_stop_steps']<20 or review['final_distance']>.95:
            raise ValueError('Human demonstration did not pass its recorded task review')
    if len(list(directory.glob('episode_*.npz')))!=len(records):
        raise ValueError('Unreviewed episode found beside selected human recordings')
    return {'passed':True,'episodes':len(records),'scope':'bounded human demonstration adaptation; not full-distribution or generalization acceptance',
            'manifest_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()}
