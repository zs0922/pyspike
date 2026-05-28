from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryInterface(ABC):
    @abstractmethod
    def read(self, addr: int, size: int) -> bytes:
        ...

    @abstractmethod
    def write(self, addr: int, data: bytes) -> None:
        ...


class DictMemoryInterface(MemoryInterface):
    def __init__(self):
        self._mem: dict[int, int] = {}

    def read(self, addr: int, size: int) -> bytes:
        return bytes(self._mem.get(addr + i, 0) for i in range(size))

    def write(self, addr: int, data: bytes) -> None:
        for i, b in enumerate(data):
            self._mem[addr + i] = b


class SimMemoryInterface(MemoryInterface):
    """Memory interface backed by pyspike sim_t physical memory.

    Uses sim_t.read_memory() and sim_t.write_memory() to access the
    Spike simulator's DRAM. This allows FSA's DMA engine to read/write
    the same physical memory that the RISC-V CPU accesses.
    """

    def __init__(self, sim):
        self.sim = sim

    def read(self, addr: int, size: int) -> bytes:
        return self.sim.read_memory(addr, size)

    def write(self, addr: int, data: bytes) -> None:
        ok = self.sim.write_memory(addr, data)
        if not ok:
            raise RuntimeError(
                f"SimMemoryInterface.write: sim_t.write_memory failed for "
                f"addr=0x{addr:x}, size={len(data)}"
            )


def test_sim_memory():
    mem = DictMemoryInterface()
    mem.write(0, b'\x01\x02\x03\x04')
    assert mem.read(0, 4) == b'\x01\x02\x03\x04', "DictMemoryInterface read/write mismatch"
    assert mem.read(4, 2) == b'\x00\x00', "DictMemoryInterface should return 0 for unwritten addresses"
    mem.write(2, b'\xff\xfe')
    assert mem.read(0, 4) == b'\x01\x02\xff\xfe', "DictMemoryInterface overwrite mismatch"

    print("test_sim_memory: DictMemoryInterface all passed")


def test_sim_memory_with_spike():
    try:
        from riscv.sim import sim_t
        from riscv.cfg import cfg_t, mem_cfg_t
        from riscv.debug_module import debug_module_config_t
    except ImportError:
        print("test_sim_memory_with_spike: skipped (pyspike not available)")
        return

    cfg = cfg_t(isa='rv64gc', priv='msu',
                mem_layout=[mem_cfg_t(0x80000000, 0x10000000)])
    sim = sim_t(cfg=cfg, halted=False,
                plugin_device_factories=[], args=['tests/data/libc-printf_hello.elf'],
                dm_config=debug_module_config_t())

    mem = SimMemoryInterface(sim)

    test_data = b'\xde\xad\xbe\xef'
    mem.write(0x80000000, test_data)
    readback = mem.read(0x80000000, 4)
    assert readback == test_data, f"SimMemoryInterface read/write mismatch: {readback.hex()} != {test_data.hex()}"

    cross_data = b'\x11\x22\x33\x44\x55\x66\x77\x88'
    mem.write(0x80000FFC, cross_data)
    cross_read = mem.read(0x80000FFC, 8)
    assert cross_read == cross_data, f"SimMemoryInterface cross-page mismatch: {cross_read.hex()} != {cross_data.hex()}"

    print("test_sim_memory_with_spike: all passed")


if __name__ == "__main__":
    test_sim_memory()
    test_sim_memory_with_spike()
