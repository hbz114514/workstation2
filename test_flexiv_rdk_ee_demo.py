"""Offline safety checks for flexiv_rdk_ee_demo; never connects to a robot."""

import sys
import types
import unittest
from argparse import Namespace
from contextlib import contextmanager
from unittest.mock import patch

import flexiv_rdk_ee_demo as demo
from scipy.spatial.transform import Rotation


SAFE = (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
CAMERA = (0.6, 0.8, -0.2, 0.2, 0.4, 1.0)


class FakeRobot:
    def __init__(self, serial):
        self.calls = []
        self.mode_value = "IDLE"
        self.pose = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    def fault(self):
        return False

    def mode(self):
        return self.mode_value

    def states(self):
        return types.SimpleNamespace(tcp_pose=self.pose)

    def Enable(self):
        self.calls.append("Enable")

    def operational(self):
        return True

    def SwitchMode(self, mode):
        self.calls.append("SwitchMode")
        self.mode_value = mode

    def ExecutePrimitive(self, name, params):
        self.calls.append(name)

    def Stop(self):
        self.calls.append("Stop")


class SafetyTest(unittest.TestCase):
    def args(self, execute=False):
        return Namespace(serial="Rizon4-test", confirm_serial="Rizon4-test",
                         tcp_safe_box=SAFE, fixed_camera_box=CAMERA,
                         tcp_envelope=0.2, expected_tool="Flexiv-GN01",
                         execute=execute)

    def vendor(self, robot):
        return types.SimpleNamespace(Robot=lambda serial: robot,
            Tool=lambda robot: types.SimpleNamespace(name=lambda: "Flexiv-GN01"),
            Mode=types.SimpleNamespace(IDLE="IDLE", NRT_PRIMITIVE_EXECUTION="NRT"),
            Coord=lambda *args: args)

    def test_camera_and_workspace_reject_path(self):
        start = demo.Pose((0.0, 0.0, 1.0), Rotation.identity())
        near_camera = demo.Pose((0.75, 0.0, 1.0), Rotation.identity())
        with self.assertRaisesRegex(ValueError, "camera"):
            demo.check_straight_path(start, near_camera, SAFE, CAMERA, 0.2)
        with self.assertRaisesRegex(ValueError, "allowed box"):
            demo.guarded_point((0.9, 0.0, 1.0), SAFE, CAMERA, 0.2)

    def test_preview_sends_no_motion_or_enable(self):
        robot = FakeRobot("Rizon4-test")
        with patch.dict(sys.modules, {"flexivrdk": self.vendor(robot)}), \
             patch.object(demo.time, "sleep", return_value=None):
            demo.run(self.args())
        self.assertEqual(robot.calls, [])

    def test_operator_stop_invokes_rdk_stop(self):
        robot = FakeRobot("Rizon4-test")
        with patch.dict(sys.modules, {"flexivrdk": self.vendor(robot)}), \
             patch.object(demo.sys.stdin, "isatty", return_value=True), \
             patch("builtins.input", return_value="MOVE"), \
             patch.object(demo.time, "sleep", return_value=None), \
             patch.object(demo, "wait_for_pose", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                demo.run(self.args(execute=True))
        self.assertEqual(robot.calls, ["Enable", "SwitchMode", "MoveL", "Stop"])

    def test_s_key_interrupts_motion_poll(self):
        robot = FakeRobot("Rizon4-test")
        robot.mode_value = "NRT"

        @contextmanager
        def key_input():
            yield lambda: b"s"

        target = demo.Pose((0.0, 0.0, 1.01), Rotation.identity())
        with patch.object(demo, "single_key_input", key_input):
            with self.assertRaises(KeyboardInterrupt):
                demo.wait_for_pose(robot, self.vendor(robot), target,
                                   SAFE, CAMERA, 0.2)


if __name__ == "__main__":
    unittest.main()
