import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from m20pro_vla.runtime.experiment import build_experiment_plan
from m20pro_vla.cli import command_experiment

ROOT=Path(__file__).resolve().parents[1]

class RawConversionSkipTest(unittest.TestCase):
    def config(self,root,backend):
        cfg=json.loads((ROOT/'configs/experiment.json').read_text());cfg['smolvla']['dataset_backend']=backend;cfg['paths']['lerobot_dataset']=str(root/'converted');p=root/'config.json';p.write_text(json.dumps(cfg));return p
    def test_raw_plan_has_no_conversion_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            p=self.config(Path(folder),'raw');plan=build_experiment_plan(p)
            self.assertNotIn('convert',plan['stages']);self.assertEqual(plan['stages']['train-smolvla']['dataset_backend'],'raw')
    def test_explicit_convert_request_on_raw_skips_before_environment_or_converter(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=self.config(root,'raw');args=argparse.Namespace(config=p,stage='convert',dry_run=False,json=True)
            with patch('m20pro_vla.cli._dispatch_lerobot_stage_environment',side_effect=AssertionError('unexpected dispatch')) as dispatch,patch('subprocess.run',side_effect=AssertionError('unexpected process')),contextlib.redirect_stdout(io.StringIO()) as output:
                code=command_experiment(args)
            self.assertEqual(code,0);self.assertIn('"skipped": true',output.getvalue());dispatch.assert_not_called();self.assertFalse((root/'converted').exists())
    def test_legacy_backend_keeps_explicit_conversion_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            plan=build_experiment_plan(self.config(Path(folder),'lerobot'));self.assertIn('convert',plan['stages'])

if __name__=='__main__':unittest.main()
