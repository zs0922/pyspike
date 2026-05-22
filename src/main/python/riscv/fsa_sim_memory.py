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
    """Placeholder: requires C++ bindings for pyspike sim_t memory access.

    The pyspike sim_t object does not currently expose mem_read/mem_write
    APIs to Python.  Until those bindings are implemented, this class
    raises RuntimeError on every read/write call.  Use DictMemoryInterface
    for functional testing.
    """

    def __init__(self, sim):
        self.sim = sim

    def read(self, addr: int, size: int) -> bytes:
        raise RuntimeError(
            "SimMemoryInterface.read: pyspike sim_t does not expose a "
            "memory read API.  Use DictMemoryInterface for functional testing."
        )

    def write(self, addr: int, data: bytes) -> None:
        raise RuntimeError(
            "SimMemoryInterface.write: pyspike sim_t does not expose a "
            "memory write API.  Use DictMemoryInterface for functional testing."
        )


def test_sim_memory():
    mem = DictMemoryInterface()
    mem.write(0, b'\x01\x02\x03\x04')
    assert mem.read(0, 4) == b'\x01\x02\x03\x04', "DictMemoryInterface read/write mismatch"
    assert mem.read(4, 2) == b'\x00\x00', "DictMemoryInterface should return 0 for unwritten addresses"
    mem.write(2, b'\xff\xfe')
    assert mem.read(0, 4) == b'\x01\x02\xff\xfe', "DictMemoryInterface overwrite mismatch"

    sim = SimMemoryInterface(sim=None)
    try:
        sim.read(0, 4)
        assert False, "SimMemoryInterface.read should raise RuntimeError"
    except RuntimeError:
        pass
    try:
        sim.write(0, b'\x00')
        assert False, "SimMemoryInterface.write should raise RuntimeError"
    except RuntimeError:
        pass

    print("test_sim_memory: all passed")


if __name__ == "__main__":
    test_sim_memory()
