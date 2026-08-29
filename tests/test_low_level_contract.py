import unittest

import numpy as np

from m20pro_vla.low_level import LOW_LEVEL_CONTRACT, M20BodyCommand


class LowLevelContractTest(unittest.TestCase):
    def test_body_command_clipping_is_stable(self):
        command = M20BodyCommand(2.0, -2.0, 2.0, False).clipped()
        self.assertTrue(np.allclose(command.as_array(), (0.4, -0.2, 0.15, 0.0)))

    def test_low_level_contract_is_versioned(self):
        self.assertEqual(LOW_LEVEL_CONTRACT, "m20_low_level_v1")


if __name__ == "__main__":
    unittest.main()
