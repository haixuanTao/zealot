"""Wireless remote parsing for the 40-byte `wireless_remote` field of LowState.

Byte layout follows unitree_rl_gym's deploy_real remote controller
(BSD-3-Clause, Unitree Robotics).
"""
import struct


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0.0  # left stick horizontal, right = +1
        self.ly = 0.0  # left stick vertical, up = +1
        self.rx = 0.0  # right stick horizontal
        self.ry = 0.0  # right stick vertical
        self.button = [0] * 16

    def set(self, data):
        keys = struct.unpack("H", bytes(data[2:4]))[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", bytes(data[4:8]))[0]
        self.rx = struct.unpack("f", bytes(data[8:12]))[0]
        self.ry = struct.unpack("f", bytes(data[12:16]))[0]
        self.ly = struct.unpack("f", bytes(data[20:24]))[0]
