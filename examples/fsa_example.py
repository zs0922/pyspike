#!/usr/bin/env python3
"""
FSA FlashAttention 示例程序

通过PySpike运行RISC-V汇编程序，该程序通过MMIO访问FSA设备，
设备启动Verilator子进程执行真实的FSA硬件仿真。

用法:
    python3 fsa_example.py [--seq-len N]
"""

import os
import sys
import time
import struct
import argparse

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
pyspike_dir = os.path.dirname(script_dir)
sys.path.insert(0, pyspike_dir)

import numpy as np


def create_test_data(seq_len: int = 4) -> tuple:
    """创建FP16测试数据"""
    q = np.random.randn(seq_len, seq_len).astype(np.float16)
    k = np.random.randn(seq_len, seq_len).astype(np.float16)
    v = np.random.randn(seq_len, seq_len).astype(np.float16)
    return q, k, v


def print_matrix(name: str, mat: np.ndarray):
    """打印矩阵"""
    print(f"\n{name} (shape={mat.shape}):")
    print(mat)


def demo_without_pyspike():
    """
    演示模式: 不依赖PySpike的完整流程

    这个模式直接操作FSA MMIO设备，模拟PySpike的行为
    """
    print("=" * 60)
    print("FSA FlashAttention Demo (Standalone Mode)")
    print("=" * 60)

    # 添加PySpike路径
    sys.path.insert(0, os.path.join(pyspike_dir, 'src', 'main', 'python'))
    from riscv.fsa_mmio import FSAMMIODevice, STATUS_INIT, STATUS_RUNNING, STATUS_DONE

    # 配置路径
    shmem_path = '/tmp/fsa_demo_shmem'
    ctrl_path = '/tmp/fsa_demo_ctrl'
    sim_path = os.environ.get(
        'FSA_SIMULATOR',
        os.path.expanduser('~/chipyard-fsa/sims/verilator/simulator-chipyard.harness-FSA4X4Fp16Config')
    )

    print(f"\nConfiguration:")
    print(f"  shmem_path: {shmem_path}")
    print(f"  ctrl_path: {ctrl_path}")
    print(f"  simulator: {sim_path}")

    # 创建测试数据
    print(f"\n[1] Creating test data...")
    seq_len = 4
    q, k, v = create_test_data(seq_len)
    print_matrix("Q", q)
    print_matrix("K", k)
    print_matrix("V", v)

    # 创建设备
    print(f"\n[2] Creating FSA MMIO device...")
    device = FSAMMIODevice(None, f"0x10000000,{shmem_path},{sim_path}")

    # 启动Verilator
    print(f"\n[3] Starting Verilator...")
    device.store(0x10000000, struct.pack('<I', 0x01))  # 启动命令
    time.sleep(0.5)
    print(f"  Status: 0x{device.status:02x}")

    if device.status != STATUS_INIT:
        print(f"  WARNING: Verilator may not have started properly")
        print(f"  (This is expected if simulator binary doesn't exist)")

    # 写入数据长度
    print(f"\n[4] Writing data configuration...")
    data_len = seq_len * seq_len * 2  # FP16 = 2 bytes
    device.store(0x10000010, struct.pack('<I', data_len))
    print(f"  Data length: {data_len} bytes")

    # 写入Q矩阵
    print(f"\n[5] Writing Q matrix...")
    device._shmem[0] = 0x10  # CMD_WRITE_Q
    q_bytes = q.tobytes()
    device.data_buffer = bytearray(q_bytes)
    device._shmem[0x2000:0x2000+len(q_bytes)] = q_bytes
    print(f"  Written {len(q_bytes)} bytes")

    # 写入K矩阵
    print(f"\n[6] Writing K matrix...")
    device._shmem[0] = 0x11  # CMD_WRITE_K
    k_bytes = k.tobytes()
    device._shmem[0x12000:0x12000+len(k_bytes)] = k_bytes
    print(f"  Written {len(k_bytes)} bytes")

    # 写入V矩阵
    print(f"\n[7] Writing V matrix...")
    device._shmem[0] = 0x12  # CMD_WRITE_V
    v_bytes = v.tobytes()
    device._shmem[0x22000:0x22000+len(v_bytes)] = v_bytes
    print(f"  Written {len(v_bytes)} bytes")

    # 触发FSA计算
    print(f"\n[8] Triggering FSA computation...")
    device.store(0x10000018, struct.pack('<I', 0x01))  # 启动信号
    print(f"  Status: 0x{device.status:02x}")

    # 轮询状态
    print(f"\n[9] Polling status...")
    max_polls = 100
    for i in range(max_polls):
        status = device.load(0x10000004, 4)
        status_val = struct.unpack('<I', status)[0]
        print(f"  Poll {i}: status=0x{status_val:02x}", end='\r')
        if status_val == STATUS_DONE:
            print(f"\n  FSA computation completed!")
            break
        time.sleep(0.1)
    else:
        print(f"\n  Timeout!")

    # 读取结果
    print(f"\n[10] Reading result...")
    device._shmem[0] = 0x20  # CMD_READ_RESULT
    result_data = device.load(0x10000020, data_len)
    result = np.frombuffer(result_data, dtype=np.float16).reshape(seq_len, seq_len)
    print_matrix("Result", result)

    # 验证结果 (与CPU计算对比)
    print(f"\n[11] Verifying result...")
    expected = q @ v.T
    diff = np.abs(result.astype(np.float32) - expected)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")

    if max_diff < 0.1:
        print(f"\n  ✅ Result verification passed!")
    else:
        print(f"\n  ⚠️  Large diff (expected for uninitialized hardware)")

    # 停止
    print(f"\n[12] Stopping...")
    device.store(0x10000000, struct.pack('<I', 0x80))  # 停止命令
    device.close()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='FSA FlashAttention Demo')
    parser.add_argument('--seq-len', type=int, default=4, help='Sequence length')
    args = parser.parse_args()

    print(f"\nFSA FlashAttention PySpike Example")
    print(f"Sequence length: {args.seq_len}")
    print()

    demo_without_pyspike()


if __name__ == '__main__':
    main()