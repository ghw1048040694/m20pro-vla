import unittest
from copy import deepcopy
from m20pro_vla.training.smolvla import smolvla_temporal_spec,configure_temporal_training_environment
from m20pro_vla.data.temporal_rgb import TemporalRGBSpec
class TemporalExperimentConfigTests(unittest.TestCase):
    def config(self):return {'smolvla':{'dataset_backend':'raw','source_fps':50,'frame_stride':2,'temporal_rgb':TemporalRGBSpec().to_dict()}}
    def test_experiment_spec_wins_over_inherited_shell_state(self):
        c=self.config();env=configure_temporal_training_environment(c,{'M20PRO_TEMPORAL_TRAINING_SPEC':'bad','KEEP':'yes'})
        import json
        self.assertEqual(json.loads(env['M20PRO_TEMPORAL_TRAINING_SPEC']),c['smolvla']['temporal_rgb']);self.assertEqual(env['KEEP'],'yes')
    def test_default_current_only_cannot_be_silently_changed_by_shell(self):
        c=self.config();c['smolvla'].pop('temporal_rgb');self.assertIsNone(smolvla_temporal_spec(c));self.assertNotIn('M20PRO_TEMPORAL_TRAINING_SPEC',configure_temporal_training_environment(c,{'M20PRO_TEMPORAL_TRAINING_SPEC':'inherited'}))
    def test_wrong_backend_or_source_tick_rate_rejected(self):
        for key,value in [('dataset_backend','lerobot'),('frame_stride',1)]:
            c=self.config();c['smolvla'][key]=value
            with self.assertRaises(ValueError):smolvla_temporal_spec(c)
    def test_unknown_contract_or_future_offsets_rejected(self):
        for key,value in [('schema','unknown'),('offsets',[-400,1,0])]:
            c=self.config();c['smolvla']['temporal_rgb'][key]=value
            with self.assertRaises(ValueError):smolvla_temporal_spec(c)
if __name__=='__main__':unittest.main()
