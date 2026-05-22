#!/usr/bin/env python3
import sys
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
pyspike_dir = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(pyspike_dir, 'src', 'main', 'python'))

from riscv.fsa_config import (
    FSAParams, fsa4x4,
    DMA_LD_SRAM, DMA_ST_SRAM,
    MX_LOAD_STATIONARY, MX_ATTENTION_SCORE_COMPUTE,
    MX_ATTENTION_VALUE_COMPUTE, MX_ATTENTION_LSE_NORM_SCALE,
    MX_ATTENTION_LSE_NORM,
    INST_DMA, INST_MATRIX, INST_FENCE,
    STATE_DONE,
)
from riscv.fsa_engine import FSAEngine, DictMemoryInterface
from riscv.fsa_decoder import DMAInstruction, MatrixInstruction, FenceInstruction


_INST_NAMES = {
    MX_LOAD_STATIONARY: "MX_LOAD_STATIONARY",
    MX_ATTENTION_SCORE_COMPUTE: "MX_ATTENTION_SCORE_COMPUTE",
    MX_ATTENTION_VALUE_COMPUTE: "MX_ATTENTION_VALUE_COMPUTE",
    MX_ATTENTION_LSE_NORM_SCALE: "MX_ATTENTION_LSE_NORM_SCALE",
    MX_ATTENTION_LSE_NORM: "MX_ATTENTION_LSE_NORM",
}

_DMA_FUNC_NAMES = {
    DMA_LD_SRAM: "DMA_LD_SRAM",
    DMA_ST_SRAM: "DMA_ST_SRAM",
}


def print_inst(inst, idx):
    if isinstance(inst, FenceInstruction):
        print(f"  [{idx:2d}] FENCE matrix={inst.matrix} dma={inst.dma} stop={inst.stop}")
    elif isinstance(inst, MatrixInstruction):
        name = _INST_NAMES.get(inst.func, f"MX_FUNC_{inst.func}")
        print(f"  [{idx:2d}] {name} spad_addr={inst.spad_addr} acc_addr={inst.acc_addr} zero={inst.zero}")
    elif isinstance(inst, DMAInstruction):
        name = _DMA_FUNC_NAMES.get(inst.func, f"DMA_FUNC_{inst.func}")
        accum = " accum" if inst.isAccum else ""
        print(f"  [{idx:2d}] {name} sram_addr={inst.sram_addr} mem_addr=0x{inst.mem_addr:X} "
              f"size={inst.size} repeat={inst.repeat}{accum}")


def numpy_flash_attention(Q_f32, K_f32, V_f32):
    dk = Q_f32.shape[1]
    S = Q_f32 @ K_f32.T / np.sqrt(dk)
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P = exp_S / np.sum(exp_S, axis=1, keepdims=True)
    return P @ V_f32


def main():
    params = fsa4x4()
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

    mem = DictMemoryInterface()
    for i in range(rows):
        mem.write(Q_BASE + i * 8, Q_fp16[i, :].tobytes())
    for i in range(rows):
        mem.write(K_BASE + i * 8, K_fp16[i, :].tobytes())
    for i in range(rows):
        mem.write(V_BASE + i * 8, V_fp16[i, :].tobytes())

    engine = FSAEngine(params, mem)

    print("=" * 60)
    print("FSA Standalone Demo - Instruction-Level Programming")
    print("=" * 60)
    print(f"Matrix size: {rows}x{cols} FP16")
    print(f"Params: saRows={params.saRows} saCols={params.saCols} "
          f"spadRows={params.spadRows} accRows={params.accRows}")

    instructions = [
        DMAInstruction(
            instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=DMA_LD_SRAM, repeat=4, sram_addr=3, sram_stride=1,
            isAccum=False, mem_stride1=0, mem_addr=Q_BASE, stride2=0,
            size=8, memStride=8,
        ),
        MatrixInstruction(
            instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=MX_LOAD_STATIONARY, waitPrevAcc=False,
            spad_addr=3, spad_stride=1, revInput=False, revOutput=False,
            delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
        ),
        DMAInstruction(
            instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=DMA_LD_SRAM, repeat=4, sram_addr=7, sram_stride=1,
            isAccum=False, mem_stride1=0, mem_addr=K_BASE, stride2=0,
            size=8, memStride=8,
        ),
        MatrixInstruction(
            instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=MX_ATTENTION_SCORE_COMPUTE, waitPrevAcc=False,
            spad_addr=7, spad_stride=1, revInput=False, revOutput=False,
            delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
        ),
        DMAInstruction(
            instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=DMA_LD_SRAM, repeat=4, sram_addr=11, sram_stride=1,
            isAccum=False, mem_stride1=0, mem_addr=V_BASE, stride2=0,
            size=8, memStride=8,
        ),
        MatrixInstruction(
            instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=MX_ATTENTION_VALUE_COMPUTE, waitPrevAcc=False,
            spad_addr=11, spad_stride=1, revInput=False, revOutput=False,
            delayOutput=False, acc_addr=1, acc_stride=1, zero=True,
        ),
        MatrixInstruction(
            instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=MX_ATTENTION_LSE_NORM_SCALE, waitPrevAcc=False,
            spad_addr=0, spad_stride=1, revInput=False, revOutput=False,
            delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
        ),
        MatrixInstruction(
            instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=MX_ATTENTION_LSE_NORM, waitPrevAcc=False,
            spad_addr=0, spad_stride=1, revInput=False, revOutput=False,
            delayOutput=False, acc_addr=1, acc_stride=1, zero=False,
        ),
        DMAInstruction(
            instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
            releaseValid=False, releaseSemValue=0,
            func=DMA_ST_SRAM, repeat=4, sram_addr=1, sram_stride=1,
            isAccum=True, mem_stride1=0, mem_addr=O_BASE, stride2=0,
            size=16, memStride=16,
        ),
        FenceInstruction(
            instType=INST_FENCE, matrix=True, dma=True, stop=True,
        ),
    ]

    print(f"\nInstruction sequence ({len(instructions)} instructions):")
    for i, inst in enumerate(instructions):
        print_inst(inst, i)

    for inst in instructions:
        engine.inst_queue.append(inst)

    print(f"\nExecuting...")
    engine.execute()

    assert engine.state == STATE_DONE, f"Expected STATE_DONE, got {engine.state}"
    print(f"Engine state: DONE")

    print(f"\nPerformance counters:")
    print(f"  execTime:  {engine.perf.execTime}")
    print(f"  mxInst:    {engine.perf.mxInst}")
    print(f"  dmaInst:   {engine.perf.dmaInst}")
    print(f"  fence:     {engine.perf.fence}")
    print(f"  rawInst:   {engine.perf.rawInst}")

    result = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        data = mem.read(O_BASE + i * 16, 16)
        result[i, :] = np.frombuffer(data, dtype=np.float32)

    O_ref = numpy_flash_attention(
        Q_fp16.astype(np.float32),
        K_fp16.astype(np.float32),
        V_fp16.astype(np.float32),
    )

    max_err = np.max(np.abs(result - O_ref))
    mean_err = np.mean(np.abs(result - O_ref))

    print(f"\nNumerical verification:")
    print(f"  Max error:  {max_err:.6f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  PASS: {max_err < 0.01}")

    print(f"\nFSA Result:")
    print(result)
    print(f"\nNumpy Reference:")
    print(O_ref)

    print(f"\n{'='*60}")
    print(f"Demo completed! Result: {'PASS' if max_err < 0.01 else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
