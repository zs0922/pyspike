#!/usr/bin/env python3
import sys
import os
import struct
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
pyspike_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(pyspike_dir, 'src', 'main', 'python'))

from riscv.fsa_config import (
    FSAParams, fsa4x4, fsa16x16,
    REG_INST_QUEUE, REG_SET_ACTIVE, REG_STATE,
    REG_PERF_RAW_INST, REG_PERF_MX_INST, REG_PERF_DMA_INST, REG_PERF_FENCE,
    STATE_IDLE, STATE_ACTIVE, STATE_DONE,
)
from riscv.fsa_mmio import FSAMMIO
from riscv.fsa_driver import FlashAttentionDriver
from riscv.fsa_sim_memory import DictMemoryInterface


def _read_reg(device, offset):
    return int.from_bytes(device.load(offset, 4), "little")


def _write_reg(device, offset, value):
    device.store(offset, value.to_bytes(4, "little"))


def numpy_flash_attention(Q_f32, K_f32, V_f32):
    dk = Q_f32.shape[1]
    S = Q_f32 @ K_f32.T / np.sqrt(dk)
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P = exp_S / np.sum(exp_S, axis=1, keepdims=True)
    return P @ V_f32


def load_matrices(mem, Q_fp16, K_fp16, V_fp16, q_base, k_base, v_base):
    rows, cols = Q_fp16.shape
    for i in range(rows):
        mem.write(q_base + i * (cols * 2), Q_fp16[i, :].tobytes())
    for i in range(rows):
        mem.write(k_base + i * (cols * 2), K_fp16[i, :].tobytes())
    for i in range(rows):
        mem.write(v_base + i * (cols * 2), V_fp16[i, :].tobytes())


def read_result(mem, o_base, rows, cols):
    result = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        data = mem.read(o_base + i * (cols * 4), cols * 4)
        result[i, :] = np.frombuffer(data, dtype=np.float32)
    return result


def run_demo(params, label):
    rows = params.saRows
    cols = params.saCols
    np.random.seed(42)

    Q_fp16 = np.random.randn(rows, cols).astype(np.float16)
    K_fp16 = np.random.randn(rows, cols).astype(np.float16)
    V_fp16 = np.random.randn(rows, cols).astype(np.float16)

    Q_BASE = 0x1000
    K_BASE = 0x2000
    V_BASE = 0x3000
    O_BASE = 0x4000

    spad_q = 3
    spad_k = 3 + cols
    spad_v = 3 + 2 * cols
    acc_o = 1
    acc_lse = 0

    mem = DictMemoryInterface()
    load_matrices(mem, Q_fp16, K_fp16, V_fp16, Q_BASE, K_BASE, V_BASE)

    device = FSAMMIO(params=params, memory=mem)
    print(f"\n{'='*60}")
    print(f"FSA FlashAttention Demo - {label}")
    print(f"{'='*60}")
    print(f"Matrix size: {rows}x{cols} FP16")
    print(f"Initial state: {_read_reg(device, REG_STATE)} (IDLE={STATE_IDLE})")

    driver = FlashAttentionDriver(params)
    words = driver.flash_attention(
        q_addr=Q_BASE, k_addr=K_BASE, v_addr=V_BASE, o_addr=O_BASE,
        spad_q=spad_q, spad_k=spad_k, spad_v=spad_v,
        acc_o=acc_o, acc_lse=acc_lse,
    )
    print(f"Instruction words: {len(words)}")

    print(f"\n[1] Writing {len(words)} instruction words to REG_INST_QUEUE...")
    for w in words:
        _write_reg(device, REG_INST_QUEUE, w)
    print(f"    Enqueued count: {_read_reg(device, 0x2C)}")

    print(f"\n[2] Writing 1 to REG_SET_ACTIVE...")
    _write_reg(device, REG_SET_ACTIVE, 1)

    state = _read_reg(device, REG_STATE)
    print(f"    State after activation: {state} (DONE={STATE_DONE})")

    print(f"\n[3] Performance counters:")
    print(f"    RAW_INST:   {_read_reg(device, REG_PERF_RAW_INST)}")
    print(f"    MX_INST:    {_read_reg(device, REG_PERF_MX_INST)}")
    print(f"    DMA_INST:   {_read_reg(device, REG_PERF_DMA_INST)}")
    print(f"    FENCE:      {_read_reg(device, REG_PERF_FENCE)}")

    result = read_result(mem, O_BASE, rows, cols)

    O_ref = numpy_flash_attention(
        Q_fp16.astype(np.float32),
        K_fp16.astype(np.float32),
        V_fp16.astype(np.float32),
    )

    max_err = np.max(np.abs(result - O_ref))
    mean_err = np.mean(np.abs(result - O_ref))

    print(f"\n[4] Numerical verification:")
    print(f"    Max error:  {max_err:.6f}")
    print(f"    Mean error: {mean_err:.6f}")
    print(f"    PASS: {max_err < 0.01}")

    print(f"\n[5] Result (FSA):")
    print(result)
    print(f"\n[6] Reference (numpy):")
    print(O_ref)

    return max_err < 0.01


def main():
    print("FSA PySpike Demo - Hardware-Aligned Interface")
    print("Standalone mode (no pyspike required)")

    ok_4x4 = run_demo(fsa4x4(), "4x4 FP16")

    ok_16x16 = run_demo(fsa16x16(), "16x16 FP16")

    print(f"\n{'='*60}")
    print(f"Summary: 4x4={'PASS' if ok_4x4 else 'FAIL'}, 16x16={'PASS' if ok_16x16 else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
