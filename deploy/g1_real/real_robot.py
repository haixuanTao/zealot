"""DDS I/O for the physical G1 via unitree_sdk2py (hg message set)."""
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_,
    unitree_hg_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from common.remote_controller import RemoteController


class MotorMode:
    PR = 0  # series control of ankle pitch/roll (what the policies expect)
    AB = 1


class RealRobotIO:
    def __init__(self, config, net_interface: str):
        ChannelFactoryInitialize(0, net_interface)
        self.crc = CRC()
        self.remote = RemoteController()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_machine = 0
        self._last_state_time = None

        self.publisher = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        self.subscriber.Init(self._on_low_state, 10)

        print("Waiting for LowState", end="", flush=True)
        while self.low_state.tick == 0:
            print(".", end="", flush=True)
            time.sleep(0.2)
        print(" connected.")

    def _on_low_state(self, msg: LowState_):
        self.low_state = msg
        self.mode_machine = msg.mode_machine
        self.remote.set(msg.wireless_remote)
        self._last_state_time = time.perf_counter()

    def state_age(self) -> float:
        if self._last_state_time is None:
            return float("inf")
        return time.perf_counter() - self._last_state_time

    def new_cmd(self):
        cmd = unitree_hg_msg_dds__LowCmd_()
        cmd.mode_machine = self.mode_machine
        cmd.mode_pr = MotorMode.PR
        for mc in cmd.motor_cmd:
            mc.mode = 1  # enable
            mc.q = 0.0
            mc.qd = 0.0
            mc.kp = 0.0
            mc.kd = 0.0
            mc.tau = 0.0
        return cmd

    def send(self, cmd):
        cmd.mode_machine = self.mode_machine
        cmd.crc = self.crc.Crc(cmd)
        self.publisher.Write(cmd)
