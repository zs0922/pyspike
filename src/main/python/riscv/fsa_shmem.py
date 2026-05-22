"""
DEPRECATED: This module is no longer the primary IPC mechanism.

The shared-memory Verilator IPC approach has been superseded by the
MMIO-based FSA engine (riscv.fsa_engine).  Use riscv.fsa_engine.FSAEngine
and riscv.fsa_mmio.FSAMMIO instead.

This file is retained for reference only and may be removed in a future
release.
"""

import warnings

warnings.warn(
    "fsa_shmem is deprecated; use riscv.fsa_engine / riscv.fsa_mmio instead.",
    DeprecationWarning,
    stacklevel=2,
)

"""
FSA 共享内存封装 (Python 实现)

用于 PySpike 和 Verilator 之间的进程间通信
"""

import mmap
import struct
import os
import time
from ctypes import Structure, c_uint32, c_uint64

# 常量
FSA_SHMEM_SIZE = 256 * 1024
FSA_SHMEM_MAGIC = 0x4645A001  # 魔数，用于验证共享内存有效性

FSA_CTRL_OFFSET = 0x0000
FSA_INST_OFFSET = 0x1000
FSA_DATA_OFFSET = 0x2000

# 控制命令
FSA_CTRL_RESET = 0x01
FSA_CTRL_LOAD_INST = 0x02
FSA_CTRL_LOAD_MEM = 0x04
FSA_CTRL_START = 0x08
FSA_CTRL_STOP = 0x10
FSA_CTRL_READ_RESULT = 0x20
FSA_CTRL_ACK = 0x40
FSA_CTRL_ERROR = 0x80

# 状态
FSA_STATUS_IDLE = 0x00
FSA_STATUS_LOADING = 0x01
FSA_STATUS_RUNNING = 0x02
FSA_STATUS_DONE = 0x03
FSA_STATUS_ERROR = 0x04


class FSAControlHeader(Structure):
    """控制头结构 (对应 C 语言的 fsa_ctrl_header_t)"""
    _fields_ = [
        ("magic", c_uint32),        # 0x00
        ("version", c_uint32),      # 0x04
        ("control", c_uint32),      # 0x08
        ("status", c_uint32),       # 0x0C
        ("inst_count", c_uint32),   # 0x10
        ("inst_offset", c_uint32),  # 0x14
        ("mem_addr", c_uint64),     # 0x18
        ("mem_size", c_uint32),     # 0x20
        ("q_size", c_uint32),       # 0x24
        ("k_size", c_uint32),       # 0x28
        ("v_size", c_uint32),       # 0x2C
        ("result_addr", c_uint64),  # 0x30
        ("result_size", c_uint32),  # 0x38
        ("result_offset", c_uint32),# 0x3C
        ("cycles", c_uint64),       # 0x40
        ("start_time", c_uint64),   # 0x48
        ("end_time", c_uint64),     # 0x50
        ("error_code", c_uint32),   # 0x58
        ("padding1", c_uint32),     # 0x5C
        ("debug0", c_uint32),       # 0x60
        ("debug1", c_uint32),       # 0x64
        ("debug2", c_uint32),       # 0x68
        ("debug3", c_uint32),       # 0x6C
        ("debug4", c_uint32),       # 0x70
        ("debug5", c_uint32),       # 0x74
        ("debug6", c_uint32),       # 0x78
        ("debug7", c_uint32),       # 0x7C
        ("timestamp", c_uint64),    # 0x80
    ]
    _pack_ = 1


class FSASharedMemory:
    """FSA 共享内存封装类"""

    def __init__(self, path: str, create: bool = False, size: int = FSA_SHMEM_SIZE):
        """
        初始化共享内存

        Args:
            path: 共享内存文件路径
            create: 是否创建新的共享内存
            size: 共享内存大小
        """
        self.path = path
        self.size = size
        self.fd = None
        self.mmap_obj = None
        self.header = None

        self._open(create)
        print(f"[FSASharedMemory] Opened shared memory: {path} (valid: {self.is_valid()})")

    def _open(self, create: bool):
        """打开/创建共享内存"""
        flags = os.O_RDWR
        if create:
            flags |= os.O_CREAT | os.O_TRUNC

        self.fd = os.open(self.path, flags, 0o666)

        if create:
            os.ftruncate(self.fd, self.size)

        self.mmap_obj = mmap.mmap(
            self.fd,
            self.size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE
        )

        # 映射控制头
        self.header = FSAControlHeader.from_buffer(self.mmap_obj)

        if create:
            self._init_header()

    def _init_header(self):
        """初始化控制头"""
        self.header.magic = FSA_SHMEM_MAGIC
        self.header.version = 1
        self.header.control = 0
        self.header.status = FSA_STATUS_IDLE
        self.header.inst_count = 0
        self.header.inst_offset = 0
        self.header.mem_size = 0
        self.header.q_size = 0
        self.header.k_size = 0
        self.header.v_size = 0
        self.header.result_size = 0
        self.header.result_offset = 0
        self.header.error_code = 0
        self.header.cycles = 0

    def is_valid(self) -> bool:
        """检查魔数"""
        return self.header.magic == FSA_SHMEM_MAGIC

    def reset(self):
        """发送复位命令"""
        self.header.control = FSA_CTRL_RESET
        self.header.status = FSA_STATUS_IDLE
        print("[FSASharedMemory] Reset command sent")

    def load_instructions(self, inst_list: list):
        """
        加载指令到共享内存

        Args:
            inst_list: 指令列表 (每条指令为 32 位整数)
        """
        # 写入指令到指令队列
        inst_data = bytearray()
        for inst in inst_list:
            inst_data.extend(struct.pack('<I', inst))

        # 写入共享内存
        self.mmap_obj.seek(FSA_INST_OFFSET)
        self.mmap_obj.write(inst_data)

        # 更新控制头
        self.header.inst_count = len(inst_list)
        self.header.control |= FSA_CTRL_LOAD_INST
        print(f"[FSASharedMemory] Loaded {len(inst_list)} instructions")

    def load_memory(self, q_data: bytes, k_data: bytes, v_data: bytes):
        """
        加载 Q/K/V 矩阵数据到共享内存

        Args:
            q_data: Q 矩阵数据
            k_data: K 矩阵数据
            v_data: V 矩阵数据
        """
        # Q 矩阵
        self.mmap_obj.seek(FSA_DATA_OFFSET)
        self.mmap_obj.write(q_data)
        self.header.q_size = len(q_data)

        # K 矩阵
        k_offset = FSA_DATA_OFFSET + len(q_data)
        self.mmap_obj.seek(k_offset)
        self.mmap_obj.write(k_data)
        self.header.k_size = len(k_data)

        # V 矩阵
        v_offset = k_offset + len(k_data)
        self.mmap_obj.seek(v_offset)
        self.mmap_obj.write(v_data)
        self.header.v_size = len(v_data)

        # 更新内存大小
        self.header.mem_size = len(q_data) + len(k_data) + len(v_data)
        self.header.control |= FSA_CTRL_LOAD_MEM
        print(f"[FSASharedMemory] Loaded memory: Q={len(q_data)}, K={len(k_data)}, V={len(v_data)}")

    def start(self):
        """发送启动命令"""
        self.header.control |= FSA_CTRL_START
        self.header.status = FSA_STATUS_RUNNING
        print("[FSASharedMemory] Start command sent")

    def stop(self):
        """发送停止命令"""
        self.header.control |= FSA_CTRL_STOP
        self.header.status = FSA_STATUS_IDLE

    def wait_done(self, timeout_ms: int = 10000) -> bool:
        """
        等待仿真完成

        Args:
            timeout_ms: 超时时间（毫秒）

        Returns:
            True: 仿真成功完成

        Raises:
            TimeoutError: 超时
            RuntimeError: 仿真出错
        """
        start = time.time()

        while True:
            status = self.header.status

            if status == FSA_STATUS_DONE:
                elapsed = (time.time() - start) * 1000
                print(f"[FSASharedMemory] Done! Cycles: {self.header.cycles}, Time: {elapsed:.1f}ms")
                return True

            if status == FSA_STATUS_ERROR:
                raise RuntimeError(f"FSA error: code={self.header.error_code}")

            if time.time() - start > timeout_ms / 1000:
                raise TimeoutError(f"FSA timeout after {timeout_ms}ms (status={status})")

            # 轮询间隔
            time.sleep(0.001)  # 1ms

    def get_status(self) -> int:
        """获取当前状态"""
        return self.header.status

    def get_cycles(self) -> int:
        """获取仿真周期数"""
        return self.header.cycles

    def get_error_code(self) -> int:
        """获取错误代码"""
        return self.header.error_code

    def get_result(self, size: int = None) -> bytes:
        """
        读取仿真结果

        Args:
            size: 结果大小，如果为 None 则使用 header 中的值

        Returns:
            结果数据
        """
        if size is None:
            size = self.header.result_size

        result_offset = FSA_DATA_OFFSET + self.header.result_offset
        self.mmap_obj.seek(result_offset)
        return self.mmap_obj.read(size)

    def write_data(self, offset: int, data: bytes):
        """
        写入数据到数据缓冲区

        Args:
            offset: 偏移量（相对于数据区起始地址）
            data: 要写入的数据
        """
        self.mmap_obj.seek(FSA_DATA_OFFSET + offset)
        self.mmap_obj.write(data)

    def read_data(self, offset: int, size: int) -> bytes:
        """
        读取数据缓冲区的数据

        Args:
            offset: 偏移量（相对于数据区起始地址）
            size: 要读取的字节数

        Returns:
            读取的数据
        """
        self.mmap_obj.seek(FSA_DATA_OFFSET + offset)
        return self.mmap_obj.read(size)

    def read_inst(self, index: int) -> int:
        """
        读取指令队列中的指令

        Args:
            index: 指令索引

        Returns:
            指令值
        """
        offset = FSA_INST_OFFSET + index * 4
        self.mmap_obj.seek(offset)
        data = self.mmap_obj.read(4)
        return struct.unpack('<I', data)[0]

    def dump_info(self):
        """打印共享内存状态"""
        print("=" * 50)
        print("FSA Shared Memory Info:")
        print(f"  Magic:     0x{self.header.magic:08x}")
        print(f"  Version:   {self.header.version}")
        print(f"  Control:   0x{self.header.control:08x}")
        print(f"  Status:    {self.header.status}")
        print(f"  Inst Count:{self.header.inst_count}")
        print(f"  Cycles:    {self.header.cycles}")
        print(f"  Error:     {self.header.error_code}")
        print("=" * 50)

    def close(self):
        """关闭共享内存"""
        # 先释放 header 引用（它指向 mmap_obj 的缓冲区）
        self.header = None

        if self.mmap_obj:
            self.mmap_obj.close()
            self.mmap_obj = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        print(f"[FSASharedMemory] Closed: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def test_shared_memory():
    """测试共享内存"""
    import tempfile

    # 创建临时文件
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name

    try:
        # 测试创建
        print("\n[Test 1] Create shared memory")
        with FSASharedMemory(path, create=True) as shmem:
            assert shmem.is_valid(), "Magic number mismatch!"
            print("  ✓ Magic number valid")

            # 测试写入指令
            print("\n[Test 2] Load instructions")
            shmem.load_instructions([0x3c000000, 0x24200000, 0x00000000, 0x00000000])
            assert shmem.header.inst_count == 4, "Instruction count mismatch!"
            print(f"  ✓ Loaded {shmem.header.inst_count} instructions")

            # 验证指令
            print("\n[Test 3] Verify instructions")
            for i in range(4):
                inst = shmem.read_inst(i)
                print(f"  Inst[{i}] = 0x{inst:08x}")

            # 测试状态更新
            print("\n[Test 4] Status update")
            shmem.header.status = FSA_STATUS_RUNNING
            assert shmem.get_status() == FSA_STATUS_RUNNING
            print("  ✓ Status updated")

            shmem.header.status = FSA_STATUS_DONE
            shmem.header.cycles = 12345
            print(f"  ✓ Done, cycles = {shmem.get_cycles()}")

            # 测试数据写入
            print("\n[Test 5] Data buffer")
            test_data = b'\x01\x02\x03\x04\x05\x06\x07\x08'
            shmem.write_data(0, test_data)
            read_data = shmem.read_data(0, len(test_data))
            assert read_data == test_data, "Data mismatch!"
            print("  ✓ Data write/read OK")

        # 测试重新打开
        print("\n[Test 6] Reopen existing shared memory")
        with FSASharedMemory(path, create=False) as shmem:
            assert shmem.is_valid(), "Magic number mismatch after reopen!"
            assert shmem.header.inst_count == 4, "Instruction count mismatch after reopen!"
            print("  ✓ Reopen OK, instructions preserved")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    finally:
        # 清理
        if os.path.exists(path):
            os.unlink(path)


if __name__ == '__main__':
    test_shared_memory()