import sys
import os
import struct
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'main', 'python'))

from riscv.fsa_config import (
    FSAParams, fsa4x4, fsa8x8, fsa16x16,
    REG_INST_QUEUE, REG_SET_ACTIVE, REG_STATE,
    REG_PERF_EXEC_TIME, REG_PERF_MX_BUBBLE, REG_PERF_MX_ACTIVE,
    REG_PERF_DMA_ACTIVE, REG_PERF_RAW_INST, REG_PERF_MX_INST,
    REG_PERF_DMA_INST, REG_PERF_FENCE, REG_ENQ_INST_CNT, REG_DEQ_INST_CNT,
    STATE_IDLE, STATE_ACTIVE, STATE_DONE,
)
from riscv.fsa_mmio import FSAMMIO
from riscv.fsa_driver import FlashAttentionDriver
from riscv.fsa_engine import DictMemoryInterface


def _read_reg(device, offset):
    return int.from_bytes(device.load(offset, 4), "little")


def _write_reg(device, offset, value):
    device.store(offset, value.to_bytes(4, "little"))


def _numpy_flash_attention(Q_f32, K_f32, V_f32):
    dk = Q_f32.shape[1]
    S = Q_f32 @ K_f32.T / np.sqrt(dk)
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P = exp_S / np.sum(exp_S, axis=1, keepdims=True)
    return P @ V_f32


def _load_matrices_into_memory(mem, Q_fp16, K_fp16, V_fp16, q_base, k_base, v_base):
    rows = Q_fp16.shape[0]
    cols = Q_fp16.shape[1]
    for i in range(rows):
        mem.write(q_base + i * (cols * 2), Q_fp16[i, :].tobytes())
    for i in range(rows):
        mem.write(k_base + i * (cols * 2), K_fp16[i, :].tobytes())
    for i in range(rows):
        mem.write(v_base + i * (cols * 2), V_fp16[i, :].tobytes())


def _read_result_from_memory(mem, o_base, rows, cols):
    result = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        data = mem.read(o_base + i * (cols * 4), cols * 4)
        result[i, :] = np.frombuffer(data, dtype=np.float32)
    return result


def _run_flash_attention(params, Q_fp16, K_fp16, V_fp16, q_base, k_base, v_base, o_base,
                         spad_q, spad_k, spad_v, acc_o, acc_lse):
    mem = DictMemoryInterface()
    _load_matrices_into_memory(mem, Q_fp16, K_fp16, V_fp16, q_base, k_base, v_base)

    device = FSAMMIO(params=params, memory=mem)
    assert _read_reg(device, REG_STATE) == STATE_IDLE

    driver = FlashAttentionDriver(params)
    words = driver.flash_attention(
        q_addr=q_base, k_addr=k_base, v_addr=v_base, o_addr=o_base,
        spad_q=spad_q, spad_k=spad_k, spad_v=spad_v,
        acc_o=acc_o, acc_lse=acc_lse,
    )

    for w in words:
        _write_reg(device, REG_INST_QUEUE, w)

    _write_reg(device, REG_SET_ACTIVE, 1)

    state = _read_reg(device, REG_STATE)
    poll_count = 0
    while state == STATE_ACTIVE and poll_count < 100:
        state = _read_reg(device, REG_STATE)
        poll_count += 1

    assert state == STATE_DONE, f"Expected STATE_DONE, got {state}"

    result = _read_result_from_memory(mem, o_base, params.saRows, params.saCols)
    return result, device


def test_fsa_4x4_fp16():
    params = fsa4x4()
    np.random.seed(42)
    Q_fp16 = np.random.randn(4, 4).astype(np.float16)
    K_fp16 = np.random.randn(4, 4).astype(np.float16)
    V_fp16 = np.random.randn(4, 4).astype(np.float16)

    result, device = _run_flash_attention(
        params, Q_fp16, K_fp16, V_fp16,
        q_base=0x1000, k_base=0x2000, v_base=0x3000, o_base=0x4000,
        spad_q=3, spad_k=7, spad_v=11, acc_o=1, acc_lse=0,
    )

    O_ref = _numpy_flash_attention(
        Q_fp16.astype(np.float32),
        K_fp16.astype(np.float32),
        V_fp16.astype(np.float32),
    )
    max_err = np.max(np.abs(result - O_ref))
    assert max_err < 0.01, f"4x4 FP16 max error {max_err} >= 0.01"


def test_fsa_8x8_fp16():
    params = fsa8x8()
    np.random.seed(123)
    Q_fp16 = np.random.randn(8, 8).astype(np.float16)
    K_fp16 = np.random.randn(8, 8).astype(np.float16)
    V_fp16 = np.random.randn(8, 8).astype(np.float16)

    result, device = _run_flash_attention(
        params, Q_fp16, K_fp16, V_fp16,
        q_base=0x1000, k_base=0x2000, v_base=0x3000, o_base=0x4000,
        spad_q=3, spad_k=11, spad_v=19, acc_o=1, acc_lse=0,
    )

    O_ref = _numpy_flash_attention(
        Q_fp16.astype(np.float32),
        K_fp16.astype(np.float32),
        V_fp16.astype(np.float32),
    )
    max_err = np.max(np.abs(result - O_ref))
    assert max_err < 0.01, f"8x8 FP16 max error {max_err} >= 0.01"


def test_fsa_multiple_iterations():
    params = fsa4x4()
    np.random.seed(99)
    Q_fp16 = np.random.randn(4, 4).astype(np.float16)
    K1_fp16 = np.random.randn(4, 4).astype(np.float16)
    K2_fp16 = np.random.randn(4, 4).astype(np.float16)
    V1_fp16 = np.random.randn(4, 4).astype(np.float16)
    V2_fp16 = np.random.randn(4, 4).astype(np.float16)

    Q_BASE = 0x1000
    K1_BASE = 0x2000
    K2_BASE = 0x6000
    V1_BASE = 0x3000
    V2_BASE = 0x7000
    O_BASE = 0x4000

    mem = DictMemoryInterface()
    for i in range(4):
        mem.write(Q_BASE + i * 8, Q_fp16[i, :].tobytes())
    for i in range(4):
        mem.write(K1_BASE + i * 8, K1_fp16[i, :].tobytes())
        mem.write(K2_BASE + i * 8, K2_fp16[i, :].tobytes())
        mem.write(V1_BASE + i * 8, V1_fp16[i, :].tobytes())
        mem.write(V2_BASE + i * 8, V2_fp16[i, :].tobytes())

    device = FSAMMIO(params=params, memory=mem)
    driver = FlashAttentionDriver(params)

    words = []
    words += driver.load_q(spad_addr=3, mem_addr=Q_BASE, rows=4, cols=4,
                           sem_id=0, acquire_sem=-1, release_sem=0)
    words += driver.load_stationary(spad_addr=3,
                                    sem_id=0, acquire_sem=0, release_sem=1)

    words += driver.load_k(spad_addr=7, mem_addr=K1_BASE, rows=4, cols=4,
                           sem_id=1, acquire_sem=-1, release_sem=1)
    words += driver.attention_score_compute(spad_addr=7, acc_addr=0,
                                            sem_id=1, acquire_sem=1, release_sem=2)

    words += driver.load_v(spad_addr=11, mem_addr=V1_BASE, rows=4, cols=4,
                           sem_id=2, acquire_sem=-1, release_sem=2)
    words += driver.attention_value_compute(spad_addr=11, acc_addr=1, zero=True,
                                            sem_id=2, acquire_sem=2, release_sem=3)

    words += driver.load_k(spad_addr=15, mem_addr=K2_BASE, rows=4, cols=4,
                           sem_id=3, acquire_sem=-1, release_sem=3)
    words += driver.attention_score_compute(spad_addr=15, acc_addr=0,
                                            sem_id=3, acquire_sem=3, release_sem=0)

    words += driver.load_v(spad_addr=19, mem_addr=V2_BASE, rows=4, cols=4,
                           sem_id=0, acquire_sem=-1, release_sem=0)
    words += driver.attention_value_compute(spad_addr=19, acc_addr=1, zero=False,
                                            sem_id=0, acquire_sem=0, release_sem=1)

    words += driver.attention_lse_norm_scale(acc_addr=0,
                                             sem_id=1, acquire_sem=-1, release_sem=1)
    words += driver.attention_lse_norm(acc_addr=1,
                                       sem_id=1, acquire_sem=1, release_sem=-1)

    words += driver.store_o(acc_addr=1, mem_addr=O_BASE, rows=4, cols=4,
                            sem_id=0, acquire_sem=-1, release_sem=-1)
    words += driver.fence(stop=True, matrix=True, dma=True)

    for w in words:
        _write_reg(device, REG_INST_QUEUE, w)

    _write_reg(device, REG_SET_ACTIVE, 1)
    state = _read_reg(device, REG_STATE)
    assert state == STATE_DONE, f"Expected STATE_DONE, got {state}"

    result = _read_result_from_memory(mem, O_BASE, 4, 4)

    K_full = np.concatenate([K1_fp16.astype(np.float32), K2_fp16.astype(np.float32)], axis=0)
    V_full = np.concatenate([V1_fp16.astype(np.float32), V2_fp16.astype(np.float32)], axis=0)
    O_ref = _numpy_flash_attention(Q_fp16.astype(np.float32), K_full, V_full)

    max_err = np.max(np.abs(result - O_ref))
    assert max_err < 0.01, f"Multi-iteration max error {max_err} >= 0.01"


def test_perf_counters():
    params = fsa4x4()
    np.random.seed(42)
    Q_fp16 = np.random.randn(4, 4).astype(np.float16)
    K_fp16 = np.random.randn(4, 4).astype(np.float16)
    V_fp16 = np.random.randn(4, 4).astype(np.float16)

    _, device = _run_flash_attention(
        params, Q_fp16, K_fp16, V_fp16,
        q_base=0x1000, k_base=0x2000, v_base=0x3000, o_base=0x4000,
        spad_q=3, spad_k=7, spad_v=11, acc_o=1, acc_lse=0,
    )

    assert _read_reg(device, REG_PERF_EXEC_TIME) > 0, "PERF_EXEC_TIME should be non-zero"
    assert _read_reg(device, REG_PERF_MX_ACTIVE) > 0, "PERF_MX_ACTIVE should be non-zero"
    assert _read_reg(device, REG_PERF_DMA_ACTIVE) > 0, "PERF_DMA_ACTIVE should be non-zero"
    assert _read_reg(device, REG_PERF_RAW_INST) > 0, "PERF_RAW_INST should be non-zero"
    assert _read_reg(device, REG_PERF_MX_INST) > 0, "PERF_MX_INST should be non-zero"
    assert _read_reg(device, REG_PERF_DMA_INST) > 0, "PERF_DMA_INST should be non-zero"
    assert _read_reg(device, REG_PERF_FENCE) > 0, "PERF_FENCE should be non-zero"


def test_state_transitions():
    params = fsa4x4()
    mem = DictMemoryInterface()
    device = FSAMMIO(params=params, memory=mem)

    assert _read_reg(device, REG_STATE) == STATE_IDLE

    driver = FlashAttentionDriver(params)
    np.random.seed(42)
    Q_fp16 = np.random.randn(4, 4).astype(np.float16)
    K_fp16 = np.random.randn(4, 4).astype(np.float16)
    V_fp16 = np.random.randn(4, 4).astype(np.float16)

    Q_BASE, K_BASE, V_BASE, O_BASE = 0x1000, 0x2000, 0x3000, 0x4000
    _load_matrices_into_memory(mem, Q_fp16, K_fp16, V_fp16, Q_BASE, K_BASE, V_BASE)

    words = driver.flash_attention(
        q_addr=Q_BASE, k_addr=K_BASE, v_addr=V_BASE, o_addr=O_BASE,
        spad_q=3, spad_k=7, spad_v=11, acc_o=1, acc_lse=0,
    )
    for w in words:
        _write_reg(device, REG_INST_QUEUE, w)

    _write_reg(device, REG_SET_ACTIVE, 1)
    assert _read_reg(device, REG_STATE) == STATE_DONE

    fence_words2 = driver.fence(stop=True)
    for w in fence_words2:
        _write_reg(device, REG_INST_QUEUE, w)

    _write_reg(device, REG_SET_ACTIVE, 1)
    assert _read_reg(device, REG_STATE) == STATE_DONE


def test_fence_stop():
    params = fsa4x4()
    mem = DictMemoryInterface()
    device = FSAMMIO(params=params, memory=mem)

    assert _read_reg(device, REG_STATE) == STATE_IDLE

    driver = FlashAttentionDriver(params)
    fence_words = driver.fence(stop=True, matrix=True, dma=True)
    for w in fence_words:
        _write_reg(device, REG_INST_QUEUE, w)

    _write_reg(device, REG_SET_ACTIVE, 1)
    assert _read_reg(device, REG_STATE) == STATE_DONE

    device2 = FSAMMIO(params=params, memory=DictMemoryInterface())
    fence_no_stop = driver.fence(stop=False, matrix=True, dma=True)
    for w in fence_no_stop:
        _write_reg(device2, REG_INST_QUEUE, w)

    _write_reg(device2, REG_SET_ACTIVE, 1)
    assert _read_reg(device2, REG_STATE) == STATE_ACTIVE
