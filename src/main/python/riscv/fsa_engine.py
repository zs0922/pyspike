from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Union

import numpy as np

from riscv.fsa_config import (
    DMA_LD_SRAM,
    DMA_ST_SRAM,
    FSAParams,
    INST_DMA,
    INST_FENCE,
    INST_MATRIX,
    MX_ATTENTION_LSE_NORM,
    MX_ATTENTION_LSE_NORM_SCALE,
    MX_ATTENTION_SCORE_COMPUTE,
    MX_ATTENTION_VALUE_COMPUTE,
    MX_LOAD_STATIONARY,
    STATE_ACTIVE,
    STATE_DONE,
    STATE_IDLE,
)
from riscv.fsa_decoder import (
    DMAInstruction,
    FenceInstruction,
    MatrixInstruction,
)
from riscv.fsa_memory import (
    Accumulator,
    FSAConfig,
    Scratchpad,
    Semaphores,
)


from riscv.fsa_sim_memory import MemoryInterface, DictMemoryInterface


@dataclass
class PerfCounters:
    execTime: int = 0
    mxBubble: int = 0
    mxActive: int = 0
    dmaActive: int = 0
    rawInst: int = 0
    mxInst: int = 0
    dmaInst: int = 0
    fence: int = 0


class FSAEngine:
    MAX_SEMAPHORE_RETRIES = 64

    def __init__(self, params: FSAParams, memory: MemoryInterface,
                 elemWidth: int = 16, accElemWidth: int = 32, beatBytes: int = 4):
        self.params = params
        self.memory = memory
        self.elemWidth = elemWidth
        self.accElemWidth = accElemWidth
        self.beatBytes = beatBytes

        cfg = FSAConfig(params, elemWidth, accElemWidth, beatBytes)
        self.spad: Scratchpad = cfg.create_scratchpad()
        self.acc: Accumulator = cfg.create_accumulator()
        self.sems: Semaphores = cfg.create_semaphores()

        self.inst_queue: Deque[Union[FenceInstruction, MatrixInstruction, DMAInstruction]] = deque()
        self.state: int = STATE_IDLE
        self.perf = PerfCounters()

        self.sa_registers: Optional[np.ndarray] = None
        self.attention_weights: Optional[np.ndarray] = None
        self.lse_scale: Optional[np.ndarray] = None
        self.row_max: Optional[np.ndarray] = None
        self.exp_sum: Optional[np.ndarray] = None

    def _acquire_semaphore(self, inst: Union[MatrixInstruction, DMAInstruction]) -> bool:
        if inst.acquireValid:
            return self.sems.acquire(inst.semId, inst.acquireSemValue)
        return True

    def _release_semaphore(self, inst: Union[MatrixInstruction, DMAInstruction]) -> None:
        if inst.releaseValid:
            self.sems.release(inst.semId, inst.releaseSemValue)

    def _execute_dma(self, inst: DMAInstruction, _retry_count: int = 0) -> None:
        self.perf.dmaInst += 1
        self.perf.dmaActive += 1

        if not self._acquire_semaphore(inst):
            if _retry_count >= self.MAX_SEMAPHORE_RETRIES:
                raise RuntimeError(f"DMA semaphore acquire failed after {self.MAX_SEMAPHORE_RETRIES} retries for semId={inst.semId}")
            self.inst_queue.appendleft(inst)
            return

        mem_addr = inst.mem_addr
        sram_addr = inst.sram_addr

        if inst.func == DMA_LD_SRAM:
            self._dma_ld_sram(inst, mem_addr, sram_addr)
        elif inst.func == DMA_ST_SRAM:
            self._dma_st_sram(inst, mem_addr, sram_addr)
        else:
            raise ValueError(f"Unknown DMA func: {inst.func}")

        self._release_semaphore(inst)

    def _validate_dma_sram_addr(self, sram_addr: int, is_accum: bool) -> None:
        max_rows = self.acc.rows if is_accum else self.spad.rows
        if sram_addr < 0 or sram_addr >= max_rows:
            raise ValueError(f"SRAM addr {sram_addr} out of range [0, {max_rows}) for {'accumulator' if is_accum else 'scratchpad'}")

    def _validate_dma_size(self, inst: DMAInstruction) -> None:
        elem_bytes = (self.accElemWidth if inst.isAccum else self.elemWidth) // 8
        if inst.size % elem_bytes != 0:
            raise ValueError(f"DMA size {inst.size} is not a multiple of element size {elem_bytes}")

    def _dma_ld_sram(self, inst: DMAInstruction, mem_addr: int, sram_addr: int) -> None:
        self._validate_dma_size(inst)
        for rep in range(inst.repeat):
            self._validate_dma_sram_addr(sram_addr, inst.isAccum)
            data_bytes = self.memory.read(mem_addr, inst.size)
            if inst.isAccum:
                arr = np.frombuffer(data_bytes, dtype=np.float32)
                self.acc.write_row(sram_addr, arr)
            else:
                arr = np.frombuffer(data_bytes, dtype=np.float16)
                self.spad.write_row(sram_addr, arr)
            mem_addr += inst.memStride
            sram_addr += inst.sram_stride

    def _dma_st_sram(self, inst: DMAInstruction, mem_addr: int, sram_addr: int) -> None:
        self._validate_dma_size(inst)
        for rep in range(inst.repeat):
            self._validate_dma_sram_addr(sram_addr, inst.isAccum)
            if inst.isAccum:
                row = self.acc.read_row(sram_addr)
                data_bytes = row.astype(np.float32).tobytes()
            else:
                row = self.spad.read_row(sram_addr)
                data_bytes = row.astype(np.float16).tobytes()
            self.memory.write(mem_addr, data_bytes[:inst.size])
            mem_addr += inst.memStride
            sram_addr += inst.sram_stride

    def _execute_matrix(self, inst: MatrixInstruction, _retry_count: int = 0) -> None:
        self.perf.mxInst += 1
        self.perf.mxActive += 1

        if not self._acquire_semaphore(inst):
            if _retry_count >= self.MAX_SEMAPHORE_RETRIES:
                raise RuntimeError(f"Matrix semaphore acquire failed after {self.MAX_SEMAPHORE_RETRIES} retries for semId={inst.semId}")
            self.inst_queue.appendleft(inst)
            return

        if inst.func == MX_LOAD_STATIONARY:
            self._mx_load_stationary(inst)
        elif inst.func == MX_ATTENTION_SCORE_COMPUTE:
            self._mx_attention_score_compute(inst)
        elif inst.func == MX_ATTENTION_VALUE_COMPUTE:
            self._mx_attention_value_compute(inst)
        elif inst.func == MX_ATTENTION_LSE_NORM_SCALE:
            self._mx_attention_lse_norm_scale(inst)
        elif inst.func == MX_ATTENTION_LSE_NORM:
            self._mx_attention_lse_norm(inst)
        else:
            raise ValueError(f"Unknown matrix func: {inst.func}")

        self._release_semaphore(inst)

    def _mx_load_stationary(self, inst: MatrixInstruction) -> None:
        cols = self.params.saCols
        rows = self.params.saRows
        q = np.zeros((rows, cols), dtype=self.spad.elemType)
        for i in range(rows):
            q[i, :] = self.spad.read_row(inst.spad_addr + i)[:cols]
        self.sa_registers = q

    def _mx_attention_score_compute(self, inst: MatrixInstruction) -> None:
        rows = self.params.saRows
        cols = self.params.saCols
        dk = cols

        k = np.zeros((rows, cols), dtype=self.spad.elemType)
        for i in range(rows):
            k[i, :] = self.spad.read_row(inst.spad_addr + i)[:cols]

        Q = self.sa_registers.astype(np.float32)
        K = k.astype(np.float32)

        S = Q @ K.T / np.sqrt(dk)

        if self.row_max is None:
            old_max = np.full(rows, -np.inf, dtype=np.float32)
            old_exp_sum = np.zeros(rows, dtype=np.float32)
        else:
            old_max = self.row_max
            old_exp_sum = self.exp_sum

        new_max = np.maximum(old_max, np.max(S, axis=1))
        correction = np.exp(old_max - new_max)
        new_exp_sum = correction * old_exp_sum + np.sum(np.exp(S - new_max[:, None]), axis=1)

        self.row_max = new_max
        self.exp_sum = new_exp_sum
        self.attention_weights = np.exp(S - new_max[:, None])

        self.acc.set_scale(correction.astype(self.acc.elemType))

        self._correction_factor = correction

        exp_sum_row = new_exp_sum.astype(self.acc.elemType)
        self.acc.write_row(0, exp_sum_row)

    def _mx_attention_value_compute(self, inst: MatrixInstruction) -> None:
        rows = self.params.saRows
        cols = self.params.saCols

        v = np.zeros((rows, cols), dtype=self.spad.elemType)
        for i in range(rows):
            v[i, :] = self.spad.read_row(inst.spad_addr + i)[:cols]

        P = self.attention_weights.astype(np.float32)
        V = v.astype(np.float32)

        PV = P @ V

        if inst.zero:
            scale = 0.0
        else:
            scale = self._correction_factor[:, None]

        old_O = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            old_O[i, :] = self.acc.read_row(inst.acc_addr + i)[:cols]

        O_new = scale * old_O + PV

        for i in range(rows):
            self.acc.write_row(inst.acc_addr + i,
                               O_new[i, :].astype(self.acc.elemType))

    def _mx_attention_lse_norm_scale(self, inst: MatrixInstruction) -> None:
        lse = self.exp_sum.astype(np.float32)
        scale = 1.0 / lse
        self.lse_scale = scale
        self.acc.set_scale(scale.astype(self.acc.elemType))

    def _mx_attention_lse_norm(self, inst: MatrixInstruction) -> None:
        rows = self.params.saRows
        cols = self.params.saCols

        for i in range(rows):
            row = self.acc.read_row(inst.acc_addr + i)[:cols]
            normalized = (self.lse_scale[i] * row.astype(np.float32)).astype(self.acc.elemType)
            self.acc.write_row(inst.acc_addr + i, normalized)

    def _execute_fence(self, inst: FenceInstruction) -> None:
        self.perf.fence += 1
        if inst.stop:
            self.state = STATE_DONE

    def execute(self) -> None:
        if self.state == STATE_DONE:
            self.perf = PerfCounters()
        if self.state not in (STATE_IDLE, STATE_DONE):
            raise ValueError(f"Cannot execute in state {self.state}, expected STATE_IDLE or STATE_DONE")
        self.state = STATE_ACTIVE
        while self.state == STATE_ACTIVE and self.inst_queue:
            inst = self.inst_queue.popleft()
            self.perf.rawInst += 1
            self.perf.execTime += 1

            if isinstance(inst, FenceInstruction):
                self._execute_fence(inst)
            elif isinstance(inst, MatrixInstruction):
                self._execute_matrix(inst)
            elif isinstance(inst, DMAInstruction):
                self._execute_dma(inst)
            else:
                raise ValueError(f"Unknown instruction type: {type(inst)}")

            if not isinstance(inst, MatrixInstruction):
                self.perf.mxBubble += 1


def test_engine():
    params = FSAParams(saRows=4, saCols=4, spadRows=24, accRows=5)
    mem = DictMemoryInterface()

    np.random.seed(42)
    Q_fp16 = np.random.randn(4, 4).astype(np.float16)
    K_fp16 = np.random.randn(4, 4).astype(np.float16)
    V_fp16 = np.random.randn(4, 4).astype(np.float16)

    Q_BASE = 0x1000
    K_BASE = 0x2000
    V_BASE = 0x3000
    O_BASE = 0x4000

    for i in range(4):
        mem.write(Q_BASE + i * 8, Q_fp16[i, :].tobytes())
        mem.write(K_BASE + i * 8, K_fp16[i, :].tobytes())
        mem.write(V_BASE + i * 8, V_fp16[i, :].tobytes())

    engine = FSAEngine(params, mem, elemWidth=16, accElemWidth=32, beatBytes=4)

    engine.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=3, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=Q_BASE, stride2=0,
        size=8, memStride=8,
    ))

    engine.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_LOAD_STATIONARY, waitPrevAcc=False,
        spad_addr=3, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
    ))

    engine.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=7, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=K_BASE, stride2=0,
        size=8, memStride=8,
    ))

    engine.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_SCORE_COMPUTE, waitPrevAcc=False,
        spad_addr=7, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
    ))

    engine.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=11, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=V_BASE, stride2=0,
        size=8, memStride=8,
    ))

    engine.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_VALUE_COMPUTE, waitPrevAcc=False,
        spad_addr=11, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=1, acc_stride=1, zero=True,
    ))

    engine.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_LSE_NORM_SCALE, waitPrevAcc=False,
        spad_addr=0, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
    ))

    engine.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_LSE_NORM, waitPrevAcc=False,
        spad_addr=0, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=1, acc_stride=1, zero=False,
    ))

    engine.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_ST_SRAM, repeat=4, sram_addr=1, sram_stride=1,
        isAccum=True, mem_stride1=0, mem_addr=O_BASE, stride2=0,
        size=16, memStride=16,
    ))

    engine.inst_queue.append(FenceInstruction(
        instType=INST_FENCE, matrix=True, dma=True, stop=True,
    ))

    engine.execute()

    assert engine.state == STATE_DONE, f"Expected STATE_DONE, got {engine.state}"

    O_result = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        data = mem.read(O_BASE + i * 16, 16)
        O_result[i, :] = np.frombuffer(data, dtype=np.float32)

    Q_f32 = Q_fp16.astype(np.float32)
    K_f32 = K_fp16.astype(np.float32)
    V_f32 = V_fp16.astype(np.float32)

    dk = 4
    S_ref = Q_f32 @ K_f32.T / np.sqrt(dk)
    S_ref_max = np.max(S_ref, axis=1, keepdims=True)
    exp_S_ref = np.exp(S_ref - S_ref_max)
    P_ref = exp_S_ref / np.sum(exp_S_ref, axis=1, keepdims=True)
    O_ref = P_ref @ V_f32

    max_err = np.max(np.abs(O_result - O_ref))
    print(f"Single-block max absolute error: {max_err:.6f}")
    assert max_err < 0.01, f"Max error {max_err} exceeds tolerance 0.01"

    print("Testing multi-block online softmax (2 K/V blocks)...")
    K1_fp16 = np.random.randn(4, 4).astype(np.float16)
    K2_fp16 = np.random.randn(4, 4).astype(np.float16)
    V1_fp16 = np.random.randn(4, 4).astype(np.float16)
    V2_fp16 = np.random.randn(4, 4).astype(np.float16)

    K1_BASE = 0x5000
    K2_BASE = 0x6000
    V1_BASE = 0x7000
    V2_BASE = 0x8000
    O2_BASE = 0x9000

    for i in range(4):
        mem.write(K1_BASE + i * 8, K1_fp16[i, :].tobytes())
        mem.write(K2_BASE + i * 8, K2_fp16[i, :].tobytes())
        mem.write(V1_BASE + i * 8, V1_fp16[i, :].tobytes())
        mem.write(V2_BASE + i * 8, V2_fp16[i, :].tobytes())

    engine2 = FSAEngine(params, mem, elemWidth=16, accElemWidth=32, beatBytes=4)

    engine2.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=3, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=Q_BASE, stride2=0,
        size=8, memStride=8,
    ))
    engine2.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_LOAD_STATIONARY, waitPrevAcc=False,
        spad_addr=3, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
    ))

    engine2.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=7, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=K1_BASE, stride2=0,
        size=8, memStride=8,
    ))
    engine2.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_SCORE_COMPUTE, waitPrevAcc=False,
        spad_addr=7, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
    ))

    engine2.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=11, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=V1_BASE, stride2=0,
        size=8, memStride=8,
    ))
    engine2.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_VALUE_COMPUTE, waitPrevAcc=False,
        spad_addr=11, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=1, acc_stride=1, zero=True,
    ))

    engine2.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=15, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=K2_BASE, stride2=0,
        size=8, memStride=8,
    ))
    engine2.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_SCORE_COMPUTE, waitPrevAcc=False,
        spad_addr=15, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
    ))

    engine2.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_LD_SRAM, repeat=4, sram_addr=19, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=V2_BASE, stride2=0,
        size=8, memStride=8,
    ))
    engine2.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_VALUE_COMPUTE, waitPrevAcc=False,
        spad_addr=19, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=1, acc_stride=1, zero=False,
    ))

    engine2.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_LSE_NORM_SCALE, waitPrevAcc=False,
        spad_addr=0, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=0, acc_stride=1, zero=False,
    ))
    engine2.inst_queue.append(MatrixInstruction(
        instType=INST_MATRIX, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=MX_ATTENTION_LSE_NORM, waitPrevAcc=False,
        spad_addr=0, spad_stride=1, revInput=False, revOutput=False,
        delayOutput=False, acc_addr=1, acc_stride=1, zero=False,
    ))

    engine2.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=False, acquireSemValue=0,
        releaseValid=False, releaseSemValue=0,
        func=DMA_ST_SRAM, repeat=4, sram_addr=1, sram_stride=1,
        isAccum=True, mem_stride1=0, mem_addr=O2_BASE, stride2=0,
        size=16, memStride=16,
    ))
    engine2.inst_queue.append(FenceInstruction(
        instType=INST_FENCE, matrix=True, dma=True, stop=True,
    ))

    engine2.execute()

    assert engine2.state == STATE_DONE, f"Expected STATE_DONE, got {engine2.state}"

    O2_result = np.zeros((4, 4), dtype=np.float32)
    for i in range(4):
        data = mem.read(O2_BASE + i * 16, 16)
        O2_result[i, :] = np.frombuffer(data, dtype=np.float32)

    K1_f32 = K1_fp16.astype(np.float32)
    K2_f32 = K2_fp16.astype(np.float32)
    V1_f32 = V1_fp16.astype(np.float32)
    V2_f32 = V2_fp16.astype(np.float32)
    K_full = np.concatenate([K1_f32, K2_f32], axis=0)
    V_full = np.concatenate([V1_f32, V2_f32], axis=0)

    S2_ref = Q_f32 @ K_full.T / np.sqrt(dk)
    S2_max = np.max(S2_ref, axis=1, keepdims=True)
    exp_S2 = np.exp(S2_ref - S2_max)
    P2_ref = exp_S2 / np.sum(exp_S2, axis=1, keepdims=True)
    O2_ref = P2_ref @ V_full

    max_err2 = np.max(np.abs(O2_result - O2_ref))
    print(f"Multi-block max absolute error: {max_err2:.6f}")
    assert max_err2 < 0.01, f"Multi-block max error {max_err2} exceeds tolerance 0.01"

    print("Testing semaphore acquire/release...")
    sem_engine = FSAEngine(params, DictMemoryInterface(), elemWidth=16, accElemWidth=32, beatBytes=4)
    sem_engine.sems.release(3, 1)
    assert sem_engine.sems.acquire(3, 1), "Semaphore acquire should succeed"
    assert sem_engine.sems.is_busy(3), "Semaphore should be busy after acquire"
    sem_engine.sems.release(3, 2)
    assert not sem_engine.sems.is_busy(3), "Semaphore should not be busy after release"
    assert sem_engine.sems.get_value(3) == 2, "Semaphore value should be 2"
    assert not sem_engine.sems.acquire(3, 1), "Acquire with wrong value should fail"
    assert sem_engine.sems.acquire(3, 2), "Acquire with matching value should succeed"
    sem_engine.sems.release(3, 0)

    print("Testing semaphore retry in instruction execution...")
    retry_mem = DictMemoryInterface()
    retry_engine = FSAEngine(params, retry_mem, elemWidth=16, accElemWidth=32, beatBytes=4)
    retry_engine.sems.acquire(0, 0)
    retry_engine.inst_queue.append(DMAInstruction(
        instType=INST_DMA, semId=0, acquireValid=True, acquireSemValue=0,
        releaseValid=True, releaseSemValue=1,
        func=DMA_LD_SRAM, repeat=1, sram_addr=3, sram_stride=1,
        isAccum=False, mem_stride1=0, mem_addr=0x1000, stride2=0,
        size=8, memStride=8,
    ))
    retry_engine.sems.release(0, 0)
    retry_engine.execute()
    assert retry_engine.sems.get_value(0) == 1, "Semaphore should be released with value 1 after retry"

    print("Testing done->active re-execution with perf counter reset...")
    reset_engine = FSAEngine(params, DictMemoryInterface(), elemWidth=16, accElemWidth=32, beatBytes=4)
    reset_engine.state = STATE_DONE
    reset_engine.perf.execTime = 100
    reset_engine.perf.mxInst = 50
    reset_engine.perf.dmaInst = 30
    reset_engine.inst_queue.append(FenceInstruction(
        instType=INST_FENCE, matrix=True, dma=True, stop=True,
    ))
    reset_engine.execute()
    assert reset_engine.state == STATE_DONE, f"Expected STATE_DONE, got {reset_engine.state}"
    assert reset_engine.perf.execTime < 100, "Perf counters should have been reset on done->active transition"
    assert reset_engine.perf.mxInst < 50, "Perf counters should have been reset"
    assert reset_engine.perf.dmaInst < 30, "Perf counters should have been reset"

    print("Testing state machine error on active->execute...")
    active_engine = FSAEngine(params, DictMemoryInterface(), elemWidth=16, accElemWidth=32, beatBytes=4)
    active_engine.state = STATE_ACTIVE
    try:
        active_engine.execute()
        assert False, "Should have raised ValueError for execute in STATE_ACTIVE"
    except ValueError:
        pass

    print(f"Perf counters (single-block): {engine.perf}")
    print(f"Perf counters (multi-block): {engine2.perf}")
    print("All engine tests passed!")


if __name__ == "__main__":
    test_engine()
