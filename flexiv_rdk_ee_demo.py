



#!/usr/bin/env python3
"""Small Flexiv RDK TCP translation/rotation check for an attended console.

Preview (connects and reads state; sends no motion):
  python flexiv_rdk_ee_demo.py --serial Rizon4-... \
    --tcp-safe-box XMIN XMAX YMIN YMAX ZMIN ZMAX \
    --fixed-camera-box XMIN XMAX YMIN YMAX ZMIN ZMAX

Add --execute --confirm-serial Rizon4-... for the attended motion test.
Coordinates are metres in the Flexiv WORLD/base frame. The camera box must
contain the fixed camera AND its support. The TCP envelope must contain the
whole gripper and wrist camera through the 5-degree rotation. This simple
geometric check does not validate the paths of upstream robot links.

During a motion, press s (without Enter) or Ctrl+C to call robot.Stop(). Keep
the physical safety-rated E-stop within reach; RDK Stop is a software stop.
"""

from __future__ import annotations

import argparse
import math
import os
import select
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

from scipy.spatial.transform import Rotation


LIFT_M = 0.010
ROTATE_DEG = 5.0
LINEAR_SPEED_M_S = 0.010
ANGULAR_SPEED_DEG_S = 5.0
MOTION_TIMEOUT_S = 15.0
POLL_S = 0.05


@dataclass(frozen=True)
class Pose:
    xyz: tuple[float, float, float]
    rotation: Rotation


def box(values: list[float], name: str) -> tuple[float, ...]:
    if len(values) != 6 or not all(math.isfinite(x) for x in values):
        raise ValueError(f"{name}: supply six finite WORLD-frame values")
    if any(values[i] >= values[i + 1] for i in (0, 2, 4)):
        raise ValueError(f"{name}: each minimum must be below its maximum")
    return tuple(values)


def guarded_point(
    xyz: tuple[float, float, float], safe: tuple[float, ...],
    camera: tuple[float, ...], envelope: float,
) -> None:
    for axis, index in enumerate((0, 2, 4)):
        if not safe[index] + envelope <= xyz[axis] <= safe[index + 1] - envelope:
            raise ValueError(f"TCP envelope leaves the allowed box on axis {axis}")
    if all(camera[i] - envelope <= xyz[axis] <= camera[i + 1] + envelope
           for axis, i in enumerate((0, 2, 4))):
        raise ValueError("TCP envelope intersects the fixed camera/support box")


def check_straight_path(
    start: Pose, end: Pose, safe: tuple[float, ...],
    camera: tuple[float, ...], envelope: float,
) -> None:
    for k in range(21):
        t = k / 20
        xyz = tuple(a + (b - a) * t for a, b in zip(start.xyz, end.xyz))
        guarded_point(xyz, safe, camera, envelope)


def read_pose(robot) -> Pose:
    raw = [float(x) for x in robot.states().tcp_pose[:7]]
    if len(raw) != 7 or not all(math.isfinite(x) for x in raw):
        raise RuntimeError("RDK did not return a finite 7-value TCP pose")
    # RDK state: x, y, z, qw, qx, qy, qz; scipy: qx, qy, qz, qw.
    return Pose(tuple(raw[:3]), Rotation.from_quat(raw[4:7] + raw[3:4]))


def pose_error(actual: Pose, target: Pose) -> tuple[float, float]:
    distance = math.dist(actual.xyz, target.xyz)
    angle = (target.rotation * actual.rotation.inv()).magnitude() * 180 / math.pi
    return distance, angle


def targets(start: Pose) -> list[tuple[str, Pose]]:
    up = Pose((start.xyz[0], start.xyz[1], start.xyz[2] + LIFT_M), start.rotation)
    turned = Pose(up.xyz, up.rotation * Rotation.from_euler("z", ROTATE_DEG, degrees=True))
    return [("lift 10 mm", up), ("rotate local Z +5 deg", turned),
            ("restore rotation", up), ("return down 10 mm", start)]


@contextmanager
def single_key_input():
    if os.name == "nt":
        import msvcrt

        yield lambda: msvcrt.getch().lower() if msvcrt.kbhit() else b""
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            yield lambda: os.read(fd, 1).lower() if select.select([fd], [], [], 0)[0] else b""
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def wait_for_pose(robot, rdk, target: Pose, safe, camera, envelope: float) -> None:
    deadline = time.monotonic() + MOTION_TIMEOUT_S
    with single_key_input() as read_key:
        print("Moving: press s or Ctrl+C to STOP immediately.", flush=True)
        while time.monotonic() < deadline:
            if read_key() == b"s":
                raise KeyboardInterrupt("operator requested software stop")
            if robot.fault() or not robot.operational():
                raise RuntimeError("robot fault or loss of operational state")
            if robot.mode() != rdk.Mode.NRT_PRIMITIVE_EXECUTION:
                raise RuntimeError("robot left primitive execution mode")
            actual = read_pose(robot)
            guarded_point(actual.xyz, safe, camera, envelope)
            linear_error, angular_error = pose_error(actual, target)
            if linear_error <= 0.002 and angular_error <= 1.0:
                return
            time.sleep(POLL_S)
    raise TimeoutError("target TCP pose not reached within 15 seconds")


def run(args: argparse.Namespace) -> None:
    safe = box(args.tcp_safe_box, "--tcp-safe-box")
    camera = box(args.fixed_camera_box, "--fixed-camera-box")
    if not math.isfinite(args.tcp_envelope) or args.tcp_envelope < 0.15:
        raise ValueError("--tcp-envelope must be at least 0.15 m")
    if args.execute and (args.confirm_serial != args.serial or not sys.stdin.isatty()):
        raise ValueError("motion requires matching --confirm-serial and an attended TTY")

    import flexivrdk as rdk

    robot = rdk.Robot(args.serial)
    stop_required = False
    try:
        if robot.fault() or robot.mode() != rdk.Mode.IDLE:
            raise RuntimeError("robot must be fault-free and IDLE before this demo")
        if rdk.Tool(robot).name() != args.expected_tool:
            raise RuntimeError(f"expected tool {args.expected_tool!r} is not active")
        start = read_pose(robot)
        time.sleep(0.3)
        drift_m, drift_deg = pose_error(read_pose(robot), start)
        if drift_m > 0.002 or drift_deg > 1.0:
            raise RuntimeError("TCP moved while checking the initial pose")
        sequence = targets(start)
        previous = start
        guarded_point(start.xyz, safe, camera, args.tcp_envelope)
        for _, target in sequence:
            check_straight_path(previous, target, safe, camera, args.tcp_envelope)
            previous = target
        print(f"TCP WORLD start: {[round(x, 4) for x in start.xyz]}")
        print("Planned: " + " -> ".join(name for name, _ in sequence))
        print(f"TCP envelope radius: {args.tcp_envelope:.3f} m")
        if not args.execute:
            print("Preview complete. No enable, mode switch, or motion command sent.")
            return

        print("Check the entire arm and both cameras visually. Hold the physical E-stop.")
        if input("Type MOVE to enable and start the attended test: ").strip() != "MOVE":
            return
        stop_required = True
        robot.Enable()
        deadline = time.monotonic() + 20.0
        while not robot.operational():
            if robot.fault() or time.monotonic() > deadline:
                raise RuntimeError("robot did not become operational")
            time.sleep(0.1)
        drift_m, drift_deg = pose_error(read_pose(robot), start)
        if drift_m > 0.002 or drift_deg > 1.0:
            raise RuntimeError("TCP changed after enabling")
        robot.SwitchMode(rdk.Mode.NRT_PRIMITIVE_EXECUTION)
        previous = start
        for index, (name, target) in enumerate(sequence, 1):
            if input(f"Step {index}/4: {name}. Type MOVE to continue: ").strip() != "MOVE":
                print("Stopped before the next step.")
                break
            actual = read_pose(robot)
            if pose_error(actual, previous)[0] > 0.004:
                raise RuntimeError("TCP drifted from the expected previous pose")
            check_straight_path(actual, target, safe, camera, args.tcp_envelope)
            # Repository's Flexiv 1.x RDK primitive convention; angles are degrees.
            robot.ExecutePrimitive("MoveL", {
                "target": rdk.Coord(list(target.xyz),
                                    target.rotation.as_euler("xyz", degrees=True).tolist(),
                                    ["WORLD", "WORLD_ORIGIN"]),
                "vel": LINEAR_SPEED_M_S,
                "angVel": ANGULAR_SPEED_DEG_S,
            })
            wait_for_pose(robot, rdk, target, safe, camera, args.tcp_envelope)
            previous = target
            print(f"Step {index} complete.")
    finally:
        if stop_required:
            try:
                robot.Stop()
                print("RDK Stop() sent; verify the robot is stationary.")
            except Exception as exc:
                print(f"RDK Stop() failed: {exc}; use the physical E-stop.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--tcp-safe-box", type=float, nargs=6, required=True,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--fixed-camera-box", type=float, nargs=6, required=True,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--tcp-envelope", type=float, default=0.20,
                        help="radius enclosing the whole TCP tool and wrist camera, m")
    parser.add_argument("--expected-tool", default="Flexiv-GN01")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-serial")
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("Operator stop requested. Confirm the robot is stationary.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Demo aborted: {exc}", file=sys.stderr)
        return 1
    return 0


def request_stop(_signum, _frame) -> None:
    raise KeyboardInterrupt("termination signal")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_stop)
    raise SystemExit(main())
