#!/usr/bin/env python
"""Drive the G1's onboard locomotion controller from a USB gamepad.

Runs ON THE ROBOT, next to `run_g1_server.py --handshake`. Reads the pad and
PUSHes `remote.*` axes/buttons as JSON to the onboard runner's action port
(6004), where `serve_onboard_controller` hands them to `robot.send_action()`
-> `controller_input` -> whichever controller was negotiated. No lerobot edits.

    # terminal A (robot): the onboard controller
    python -m lerobot.robots.unitree_g1.run_g1_server --handshake
    # terminal B (laptop): agree on the controller
    python -m lerobot.robots.unitree_g1.run_g1_server \
        --handshake-client HolosomaLocomotionController --server-ip <g1>
    # terminal C (robot): steer it
    python gamepad_teleop.py

`--dry-run` prints the axes without opening ZMQ, for checking the pad alone.

Two backends, picked automatically:

  xinput  raw libusb against the 8BitDo dongle's vendor-specific interface.
          The Jetson kernel (5.15.148-tegra) ships no `xpad`, so nothing binds
          the interface and there is no /dev/input node -- but that also means
          libusb can claim it directly. Needs the udev rule in the README.
  evdev   a real /dev/input/js*-style device, i.e. a pad in DInput mode that
          usbhid+joydev picked up. Preferred when present.

SAFETY: the controller latches the last input it received, so a dead script or
a dead pad would leave the robot walking. Axes are therefore zeroed and sent on
exit, and after --timeout seconds without a fresh packet.
"""
import argparse
import json
import signal
import struct
import sys
import time

ACTION_PORT = 6004

# Positional button indices the G1 controllers expect, mirroring
# lerobot's _REMOTE_BUTTON_MAP / unitree_rl_gym's KeyMap. Index 6/7 (F1/F2)
# have no gamepad equivalent and stay 0.
BUTTON_INDEX = {
    "RB": 0, "LB": 1, "start": 2, "back": 3, "RT": 4, "LT": 5,
    "A": 8, "B": 9, "X": 10, "Y": 11,
    "up": 12, "right": 13, "down": 14, "left": 15,
}

AXIS_KEYS = ("remote.lx", "remote.ly", "remote.rx", "remote.ry")

# Microsoft's stock XInput deadzones (XINPUT_GAMEPAD_{LEFT,RIGHT}_THUMB_DEADZONE),
# normalised. Every XInput host applies these, which is why a pad whose sticks rest
# off-centre by a couple of thousand counts still reads as centred everywhere else.
DEADZONE = {
    "remote.lx": 7849 / 32767, "remote.ly": 7849 / 32767,
    "remote.rx": 8689 / 32767, "remote.ry": 8689 / 32767,
}

# 8BitDo Ultimate 2C dongle in "GAME FOR WINDOWS" (XInput) mode.
XINPUT_IDS = [(0x2F24, 0x008F)]


def neutral() -> dict:
    """A full zeroed remote frame: sticks centred, every button released."""
    frame = dict.fromkeys(AXIS_KEYS, 0.0)
    for i in range(16):
        frame[f"remote.button.{i}"] = 0.0
    return frame


class XInputPad:
    """Xbox-360-protocol pad read straight off its interrupt IN endpoint.

    The 20-byte report is the well-known wired-360 layout:
      [0]=type [1]=len [2],[3]=button bitmasks [4]=LT [5]=RT
      [6:8]=LX [8:10]=LY [10:12]=RX [12:14]=RY   (int16 LE, +32767 = right/up)
    """

    BTN0 = [("up", 0x01), ("down", 0x02), ("left", 0x04), ("right", 0x08),
            ("start", 0x10), ("back", 0x20)]
    BTN1 = [("LB", 0x01), ("RB", 0x02),
            ("A", 0x10), ("B", 0x20), ("X", 0x40), ("Y", 0x80)]

    def __init__(self):
        import usb.core
        import usb.util

        self._util = usb.util
        self.dev = None
        for vid, pid in XINPUT_IDS:
            self.dev = usb.core.find(idVendor=vid, idProduct=pid)
            if self.dev is not None:
                break
        if self.dev is None:
            raise RuntimeError("no XInput gamepad found on USB")

        intf = self.dev.get_active_configuration()[(0, 0)]
        self.intf = intf.bInterfaceNumber
        self.ep = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
            == usb.util.ENDPOINT_IN,
        )
        if self.ep is None:
            raise RuntimeError("gamepad interface has no IN endpoint")
        # The kernel has no driver for a class-0xff interface here, but detach
        # anyway so this also works on a host where xpad IS loaded.
        try:
            if self.dev.is_kernel_driver_active(self.intf):
                self.dev.detach_kernel_driver(self.intf)
        except (NotImplementedError, usb.core.USBError):
            pass
        usb.util.claim_interface(self.dev, self.intf)
        self.name = f"XInput {self.dev.idVendor:04x}:{self.dev.idProduct:04x}"

    def read(self, timeout_ms: int) -> dict | None:
        """One frame, or None if no report arrived within the timeout."""
        import usb.core

        try:
            data = self.dev.read(self.ep.bEndpointAddress, self.ep.wMaxPacketSize,
                                 timeout=timeout_ms)
        except usb.core.USBTimeoutError:
            return None
        if len(data) < 14 or data[0] != 0x00:
            return None  # not an input report (rumble/LED acks share the pipe)

        lx, ly, rx, ry = struct.unpack("<hhhh", bytes(data[6:14]))
        frame = neutral()
        frame["remote.lx"] = lx / 32767.0
        frame["remote.ly"] = ly / 32767.0
        frame["remote.rx"] = rx / 32767.0
        frame["remote.ry"] = ry / 32767.0
        for name, mask in self.BTN0:
            frame[f"remote.button.{BUTTON_INDEX[name]}"] = float(bool(data[2] & mask))
        for name, mask in self.BTN1:
            frame[f"remote.button.{BUTTON_INDEX[name]}"] = float(bool(data[3] & mask))
        # Analogue triggers reported as pressed past half travel.
        frame[f"remote.button.{BUTTON_INDEX['LT']}"] = float(data[4] > 127)
        frame[f"remote.button.{BUTTON_INDEX['RT']}"] = float(data[5] > 127)
        return frame

    def close(self):
        with_suppress = getattr(self._util, "release_interface", None)
        if with_suppress is not None:
            try:
                self._util.release_interface(self.dev, self.intf)
            except Exception:  # noqa: BLE001 - teardown must not mask the real error
                pass


class EvdevPad:
    """Pad exposed as a kernel input device (DInput mode, usbhid+joydev)."""

    # evdev axis codes -> our stick names. ABS_X/Y = left stick,
    # ABS_RX/RY (or ABS_Z/RZ on some pads) = right stick.
    AXIS_MAP = {0: "lx", 1: "ly", 3: "rx", 4: "ry", 2: "rx", 5: "ry"}
    KEY_MAP = {
        304: "A", 305: "B", 307: "X", 308: "Y",
        310: "LB", 311: "RB", 312: "LT", 313: "RT",
        314: "back", 315: "start",
    }

    def __init__(self):
        import evdev

        pads = [
            evdev.InputDevice(path)
            for path in evdev.list_devices()
        ]
        pads = [d for d in pads if evdev.ecodes.EV_ABS in d.capabilities()]
        if not pads:
            raise RuntimeError("no evdev device with absolute axes")
        self.dev = pads[0]
        self.name = f"evdev {self.dev.name}"
        self._ranges = {
            code: (info.min, info.max)
            for code, info in self.dev.capabilities().get(evdev.ecodes.EV_ABS, [])
        }
        self._frame = neutral()
        self._ecodes = evdev.ecodes

    def read(self, timeout_ms: int) -> dict | None:
        import select

        ready, _, _ = select.select([self.dev.fd], [], [], timeout_ms / 1000.0)
        if not ready:
            return None
        got = False
        for event in self.dev.read():
            if event.type == self._ecodes.EV_ABS and event.code in self.AXIS_MAP:
                lo, hi = self._ranges.get(event.code, (-32768, 32767))
                mid = (lo + hi) / 2.0
                span = max((hi - lo) / 2.0, 1.0)
                name = self.AXIS_MAP[event.code]
                value = (event.value - mid) / span
                # Kernel Y axes point down; the G1 wants up = +1.
                self._frame[f"remote.{name}"] = -value if name.endswith("y") else value
                got = True
            elif event.type == self._ecodes.EV_KEY and event.code in self.KEY_MAP:
                idx = BUTTON_INDEX[self.KEY_MAP[event.code]]
                self._frame[f"remote.button.{idx}"] = float(event.value != 0)
                got = True
        return dict(self._frame) if got else None

    def close(self):
        self.dev.close()


def open_pad(backend: str):
    """Open the requested backend, or try evdev then XInput."""
    order = {"auto": (EvdevPad, XInputPad), "evdev": (EvdevPad,), "xinput": (XInputPad,)}[backend]
    errors = []
    for cls in order:
        try:
            return cls()
        except Exception as e:  # noqa: BLE001 - report every backend's reason together
            errors.append(f"  {cls.__name__}: {type(e).__name__}: {e}")
    raise SystemExit("no gamepad available:\n" + "\n".join(errors))


def apply_shaping(frame: dict, deadzone: dict, scale: float) -> dict:
    """Deadzone, rescale and clamp the sticks; buttons pass through."""
    out = dict(frame)
    for key in AXIS_KEYS:
        dz = deadzone[key]
        v = out[key]
        v = 0.0 if abs(v) < dz else (abs(v) - dz) / (1.0 - dz) * (1 if v > 0 else -1)
        out[key] = max(-1.0, min(1.0, v * scale))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="onboard runner host (default: local)")
    p.add_argument("--port", type=int, default=ACTION_PORT, help=f"action PUSH port (default: {ACTION_PORT})")
    p.add_argument("--backend", choices=("auto", "evdev", "xinput"), default="auto")
    p.add_argument("--rate", type=float, default=50.0, help="send rate in Hz (default: 50)")
    p.add_argument("--deadzone", type=float, default=None,
                   help="override the stock XInput deadzones (0.24 left / 0.27 right) on both sticks")
    p.add_argument("--scale", type=float, default=1.0,
                   help="scale applied to every axis after the deadzone (default: 1.0)")
    p.add_argument("--timeout", type=float, default=0.5,
                   help="seconds without a pad report before zeroing the axes (default: 0.5)")
    p.add_argument("--dry-run", action="store_true", help="print axes, do not open ZMQ")
    args = p.parse_args()

    pad = open_pad(args.backend)
    print(f"gamepad: {pad.name}")

    deadzone = DEADZONE if args.deadzone is None else dict.fromkeys(AXIS_KEYS, args.deadzone)

    sock = ctx = None
    if not args.dry_run:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUSH)
        sock.setsockopt(zmq.LINGER, 200)  # let the final zero frame drain on exit
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.connect(f"tcp://{args.host}:{args.port}")
        print(f"pushing remote.* to tcp://{args.host}:{args.port} at {args.rate:.0f} Hz")
    else:
        print("dry run: not connecting to the robot")

    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    period = 1.0 / args.rate
    read_timeout_ms = max(1, int(period * 1000))
    frame = neutral()
    last_report = time.time()
    last_print = 0.0

    def send(payload: dict) -> None:
        if sock is not None:
            sock.send_string(json.dumps(payload))

    try:
        while running:
            t0 = time.time()

            fresh = pad.read(read_timeout_ms)
            if fresh is not None:
                frame = fresh
                last_report = t0
            elif t0 - last_report > args.timeout:
                # Pad went quiet (unplugged, dongle dropped): stop the robot rather
                # than let the controller keep acting on a stale command.
                if any(frame[k] for k in AXIS_KEYS):
                    print("no gamepad report; zeroing axes")
                frame = neutral()

            shaped = apply_shaping(frame, deadzone, args.scale)
            send(shaped)

            if t0 - last_print >= 0.2:
                last_print = t0
                pressed = [i for i in range(16) if shaped[f"remote.button.{i}"]]
                sys.stdout.write(
                    f"\rlx{shaped['remote.lx']:+.2f} ly{shaped['remote.ly']:+.2f} "
                    f"rx{shaped['remote.rx']:+.2f} ry{shaped['remote.ry']:+.2f} "
                    f"btn{pressed}      "
                )
                sys.stdout.flush()

            time.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        print("\nstopping: sending zeroed axes")
        for _ in range(3):  # CONFLATE keeps only the newest, so make the zero stick
            send(neutral())
            time.sleep(0.02)
        pad.close()
        if sock is not None:
            sock.close()
            ctx.term()


if __name__ == "__main__":
    main()
