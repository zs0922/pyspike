"""FSA MMIO device – register-level interface for the FlashAttention accelerator.

Usage
-----

Standalone mode (no pyspike dependency)::

    from riscv.fsa_config import FSAParams, REG_INST_QUEUE, REG_SET_ACTIVE, REG_STATE
    from riscv.fsa_mmio import FSAMMIO
    from riscv.fsa_sim_memory import DictMemoryInterface
    from riscv.fsa_encoder import encode_fence, encode_matrix, encode_dma

    params = FSAParams(saRows=4, saCols=4, spadRows=24, accRows=5)
    mem = DictMemoryInterface()
    dev = FSAMMIO(params=params, memory=mem)

    # Write instruction words to the instruction queue register
    for w in encode_fence(matrix=True, dma=True, stop=True):
        dev.store(REG_INST_QUEUE, w.to_bytes(4, "little"))

    # Kick execution
    dev.store(REG_SET_ACTIVE, (1).to_bytes(4, "little"))

    # Poll state
    state = int.from_bytes(dev.load(REG_STATE, 4), "little")

With pyspike (C++ simulator)::

    from riscv.fsa_mmio import FSAMMIO
    dev = FSAMMIO(sim=sim, args=args)   # sim is the pyspike sim_t object

High-level driver (recommended)::

    from riscv.fsa_driver import FlashAttentionDriver
    driver = FlashAttentionDriver(params)
    words = driver.flash_attention(q_addr, k_addr, v_addr, o_addr, ...)
    for w in words:
        dev.store(REG_INST_QUEUE, w.to_bytes(4, "little"))
    dev.store(REG_SET_ACTIVE, (1).to_bytes(4, "little"))

Register map (byte offsets)
---------------------------

    0x00  REG_INST_QUEUE       WO  Write 32-bit instruction words
    0x04  REG_SET_ACTIVE       WO  Write non-zero to start execution
    0x08  REG_STATE            RO  0=IDLE, 1=ACTIVE, 2=DONE
    0x0C  REG_PERF_EXEC_TIME   RO  Total execution cycles
    0x10  REG_PERF_MX_BUBBLE   RO  Matrix-unit bubble cycles
    0x14  REG_PERF_MX_ACTIVE   RO  Matrix-unit active cycles
    0x18  REG_PERF_DMA_ACTIVE  RO  DMA active cycles
    0x1C  REG_PERF_RAW_INST    RO  Total decoded instructions
    0x20  REG_PERF_MX_INST     RO  Matrix instructions executed
    0x24  REG_PERF_DMA_INST    RO  DMA instructions executed
    0x28  REG_PERF_FENCE       RO  Fence instructions executed
    0x2C  REG_ENQ_INST_CNT     RO  Words written to INST_QUEUE
    0x30  REG_DEQ_INST_CNT     RO  Instructions dequeued by engine

Instruction formats (word counts)
----------------------------------

    Fence:   1 word   [header]
    Matrix:  3 words  [header, spad, acc]
    DMA:     4 words  [header, sram, mem_lo, mem_hi]

    Use riscv.fsa_encoder to build instruction words, or
    riscv.fsa_driver.FlashAttentionDriver for a high-level API.
"""

from typing import Optional, TYPE_CHECKING

try:
    from riscv import dev
    _using_riscv_module = True
except ImportError:
    from typing import Any
    _using_riscv_module = False

    class MMIO:
        def __init__(self, sim: Any = None, args: Any = None):
            self.sim = sim
            self.args = args

    def dev_register(name: str, **kwargs):
        def decorator(cls):
            return cls
        return decorator

    class dev:
        MMIO = MMIO
        register = dev_register

from riscv.fsa_config import (
    REG_INST_QUEUE, REG_SET_ACTIVE, REG_STATE,
    REG_PERF_EXEC_TIME, REG_PERF_MX_BUBBLE, REG_PERF_MX_ACTIVE,
    REG_PERF_DMA_ACTIVE, REG_PERF_RAW_INST, REG_PERF_MX_INST,
    REG_PERF_DMA_INST, REG_PERF_FENCE, REG_ENQ_INST_CNT, REG_DEQ_INST_CNT,
    FSA_MMIO_SIZE, STATE_IDLE, STATE_ACTIVE, STATE_DONE,
    FSAParams,
)
from riscv.fsa_decoder import Decoder
from riscv.fsa_engine import FSAEngine
from riscv.fsa_sim_memory import DictMemoryInterface


_READ_REGS = {
    REG_STATE,
    REG_PERF_EXEC_TIME, REG_PERF_MX_BUBBLE, REG_PERF_MX_ACTIVE,
    REG_PERF_DMA_ACTIVE, REG_PERF_RAW_INST, REG_PERF_MX_INST,
    REG_PERF_DMA_INST, REG_PERF_FENCE,
    REG_ENQ_INST_CNT, REG_DEQ_INST_CNT,
}

_WRITE_REGS = {REG_INST_QUEUE, REG_SET_ACTIVE}


@dev.register("fsa_mmio", size=FSA_MMIO_SIZE)
class FSAMMIO(dev.MMIO if _using_riscv_module else MMIO):

    def __init__(self, sim=None, args=None, params=None, memory=None):
        if _using_riscv_module:
            super().__init__(sim, args)
        else:
            self.sim = sim
            self.args = args
        self._state = STATE_IDLE
        self._enq_inst_cnt = 0
        self._deq_inst_cnt = 0
        self._perf = {
            REG_PERF_EXEC_TIME: 0,
            REG_PERF_MX_BUBBLE: 0,
            REG_PERF_MX_ACTIVE: 0,
            REG_PERF_DMA_ACTIVE: 0,
            REG_PERF_RAW_INST: 0,
            REG_PERF_MX_INST: 0,
            REG_PERF_DMA_INST: 0,
            REG_PERF_FENCE: 0,
        }

        if params is None:
            params = FSAParams(saRows=4, saCols=4, spadRows=24, accRows=5)
        self._params = params

        if memory is None:
            if sim is not None and _using_riscv_module:
                try:
                    from riscv.fsa_sim_memory import SimMemoryInterface
                    memory = SimMemoryInterface(sim)
                except Exception:
                    memory = DictMemoryInterface()
            else:
                memory = DictMemoryInterface()
        self._memory = memory

        self._engine = FSAEngine(params, memory)
        self._decoder = Decoder()

    def store(self, addr: int, data: bytes) -> bool:
        offset = addr & 0xFF
        value = int.from_bytes(data[:4], "little") if len(data) >= 4 else int.from_bytes(data, "little")
        if offset == REG_INST_QUEUE:
            self._enq_inst_cnt = (self._enq_inst_cnt + 1) & 0xFFFFFFFF
            try:
                inst = self._decoder.feed(value & 0xFFFFFFFF)
            except ValueError:
                self._decoder.reset()
                inst = None
            if inst is not None:
                self._engine.inst_queue.append(inst)
            return True
        if offset == REG_SET_ACTIVE:
            if value != 0:
                if self._state == STATE_IDLE:
                    self._state = STATE_ACTIVE
                elif self._state == STATE_DONE:
                    self._reset_perf()
                    self._state = STATE_ACTIVE
                if self._state == STATE_ACTIVE:
                    self._execute()
            return True
        return True

    def load(self, addr: int, size: int) -> bytes:
        offset = addr & 0xFF
        if offset == REG_STATE:
            return self._state.to_bytes(size, "little")
        if offset in self._perf:
            return self._perf[offset].to_bytes(size, "little")
        if offset == REG_ENQ_INST_CNT:
            return self._enq_inst_cnt.to_bytes(size, "little")
        if offset == REG_DEQ_INST_CNT:
            return self._deq_inst_cnt.to_bytes(size, "little")
        return (0).to_bytes(size, "little")

    def _execute(self):
        n = len(self._engine.inst_queue)
        self._engine.execute()
        self._deq_inst_cnt = (self._deq_inst_cnt + n) & 0xFFFFFFFF
        self._sync_perf()
        if self._engine.state == STATE_DONE:
            self._state = STATE_DONE

    def _sync_perf(self):
        perf = self._engine.perf
        self._perf[REG_PERF_EXEC_TIME] = perf.execTime & 0xFFFFFFFF
        self._perf[REG_PERF_MX_BUBBLE] = perf.mxBubble & 0xFFFFFFFF
        self._perf[REG_PERF_MX_ACTIVE] = perf.mxActive & 0xFFFFFFFF
        self._perf[REG_PERF_DMA_ACTIVE] = perf.dmaActive & 0xFFFFFFFF
        self._perf[REG_PERF_RAW_INST] = perf.rawInst & 0xFFFFFFFF
        self._perf[REG_PERF_MX_INST] = perf.mxInst & 0xFFFFFFFF
        self._perf[REG_PERF_DMA_INST] = perf.dmaInst & 0xFFFFFFFF
        self._perf[REG_PERF_FENCE] = perf.fence & 0xFFFFFFFF

    @staticmethod
    def _inst_word_count(inst):
        from riscv.fsa_decoder import FenceInstruction, MatrixInstruction, DMAInstruction
        if isinstance(inst, FenceInstruction):
            return 1
        elif isinstance(inst, MatrixInstruction):
            return 3
        elif isinstance(inst, DMAInstruction):
            return 4
        return 0

    def _reset_perf(self):
        for k in self._perf:
            self._perf[k] = 0
        saved_queue = list(self._engine.inst_queue)
        enq_words = sum(self._inst_word_count(i) for i in saved_queue)
        self._enq_inst_cnt = enq_words
        self._deq_inst_cnt = 0
        self._engine = FSAEngine(self._params, self._memory)
        self._engine.inst_queue.extend(saved_queue)
        self._decoder = Decoder()

    def size(self) -> int:
        return FSA_MMIO_SIZE

    def tick(self, rtc_ticks: int) -> None:
        pass


def test_register_map():
    from riscv.fsa_encoder import encode_fence, encode_matrix, encode_dma
    from riscv.fsa_config import (
        DMA_LD_SRAM, MX_LOAD_STATIONARY,
    )
    errors = []

    dev = FSAMMIO()
    if dev.load(REG_STATE, 4) != STATE_IDLE.to_bytes(4, "little"):
        errors.append("initial state not IDLE")

    for reg_offset, name in [
        (REG_PERF_EXEC_TIME, "EXEC_TIME"),
        (REG_PERF_MX_BUBBLE, "MX_BUBBLE"),
        (REG_PERF_MX_ACTIVE, "MX_ACTIVE"),
        (REG_PERF_DMA_ACTIVE, "DMA_ACTIVE"),
        (REG_PERF_RAW_INST, "RAW_INST"),
        (REG_PERF_MX_INST, "MX_INST"),
        (REG_PERF_DMA_INST, "DMA_INST"),
        (REG_PERF_FENCE, "FENCE"),
    ]:
        val = int.from_bytes(dev.load(reg_offset, 4), "little")
        if val != 0:
            errors.append(f"perf counter {name} != 0 initially, got {val}")

    deq_val = int.from_bytes(dev.load(REG_DEQ_INST_CNT, 4), "little")
    if deq_val != 0:
        errors.append(f"REG_DEQ_INST_CNT != 0 initially, got {deq_val}")

    unknown_val = int.from_bytes(dev.load(0x40, 4), "little")
    if unknown_val != 0:
        errors.append(f"unknown offset should return 0, got {unknown_val}")

    fence_words = encode_fence(matrix=True, dma=True, stop=True)
    dev.store(REG_INST_QUEUE, fence_words[0].to_bytes(4, "little"))
    if dev._enq_inst_cnt != 1:
        errors.append(f"enqInstCnt != 1 after fence push, got {dev._enq_inst_cnt}")
    if len(dev._engine.inst_queue) != 1:
        errors.append(f"engine queue != 1 after fence push, got {len(dev._engine.inst_queue)}")

    mx_words = encode_matrix(func=MX_LOAD_STATIONARY, spad_addr=3, spad_stride=1, acc_addr=0, acc_stride=1)
    for w in mx_words:
        dev.store(REG_INST_QUEUE, w.to_bytes(4, "little"))
    if dev._enq_inst_cnt != 4:
        errors.append(f"enqInstCnt != 4 after fence+matrix, got {dev._enq_inst_cnt}")
    if len(dev._engine.inst_queue) != 2:
        errors.append(f"engine queue != 2 after fence+matrix, got {len(dev._engine.inst_queue)}")

    enq_val = int.from_bytes(dev.load(REG_ENQ_INST_CNT, 4), "little")
    if enq_val != 4:
        errors.append(f"REG_ENQ_INST_CNT read != 4, got {enq_val}")

    dev.store(REG_SET_ACTIVE, (1).to_bytes(4, "little"))
    state_val = int.from_bytes(dev.load(REG_STATE, 4), "little")
    if state_val != STATE_DONE:
        errors.append(f"state not DONE after SET_ACTIVE with fence(stop), got {state_val}")

    deq_after = int.from_bytes(dev.load(REG_DEQ_INST_CNT, 4), "little")
    if deq_after != 2:
        errors.append(f"REG_DEQ_INST_CNT != 2 after execution, got {deq_after}")

    raw_inst = int.from_bytes(dev.load(REG_PERF_RAW_INST, 4), "little")
    if raw_inst != 1:
        errors.append(f"REG_PERF_RAW_INST != 1 after execution (engine resets on done), got {raw_inst}")

    dev2 = FSAMMIO()
    fence_no_stop = encode_fence(matrix=True, dma=True, stop=False)
    dev2.store(REG_INST_QUEUE, fence_no_stop[0].to_bytes(4, "little"))
    dev2.store(REG_SET_ACTIVE, (1).to_bytes(4, "little"))
    state_no_stop = int.from_bytes(dev2.load(REG_STATE, 4), "little")
    if state_no_stop != STATE_ACTIVE:
        errors.append(f"state not ACTIVE after fence(no stop), got {state_no_stop}")

    dev2.store(REG_SET_ACTIVE, (0).to_bytes(4, "little"))
    if int.from_bytes(dev2.load(REG_STATE, 4), "little") != STATE_ACTIVE:
        errors.append("state changed on SET_ACTIVE=0 while active")

    dev3 = FSAMMIO()
    dev3._state = STATE_DONE
    dev3._perf[REG_PERF_EXEC_TIME] = 42
    dev3._perf[REG_PERF_MX_INST] = 99
    dev3._enq_inst_cnt = 5
    dev3._deq_inst_cnt = 3
    fence3 = encode_fence(matrix=True, dma=True, stop=True)
    dev3.store(REG_INST_QUEUE, fence3[0].to_bytes(4, "little"))
    dev3.store(REG_SET_ACTIVE, (1).to_bytes(4, "little"))
    if dev3._perf[REG_PERF_EXEC_TIME] != 1:
        errors.append(f"perf EXEC_TIME not reset+executed on done->active, got {dev3._perf[REG_PERF_EXEC_TIME]}")
    if dev3._perf[REG_PERF_FENCE] != 1:
        errors.append(f"perf FENCE not 1 on done->active, got {dev3._perf[REG_PERF_FENCE]}")
    if dev3._enq_inst_cnt != 1:
        errors.append(f"enqInstCnt not set to queue word count on done->active, expected 1 got {dev3._enq_inst_cnt}")
    if dev3._deq_inst_cnt != 1:
        errors.append(f"deqInstCnt != 1 after done->active execution, got {dev3._deq_inst_cnt}")

    dev4 = FSAMMIO()
    dev4.store(REG_SET_ACTIVE, (0).to_bytes(4, "little"))
    idle_after_zero = int.from_bytes(dev4.load(REG_STATE, 4), "little")
    if idle_after_zero != STATE_IDLE:
        errors.append(f"state changed on SET_ACTIVE=0, got {idle_after_zero}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        raise AssertionError(f"{len(errors)} test(s) failed")
    print("test_register_map: all passed")


def test_integration():
    import numpy as np
    from riscv.fsa_config import (
        DMA_LD_SRAM, DMA_ST_SRAM,
        MX_LOAD_STATIONARY, MX_ATTENTION_SCORE_COMPUTE,
        MX_ATTENTION_VALUE_COMPUTE, MX_ATTENTION_LSE_NORM_SCALE,
        MX_ATTENTION_LSE_NORM,
    )
    from riscv.fsa_encoder import encode_fence, encode_matrix, encode_dma

    errors = []

    def check(name, actual, expected):
        if actual != expected:
            errors.append(f"  {name}: expected {expected}, got {actual}")

    params = FSAParams(saRows=4, saCols=4, spadRows=24, accRows=5)
    mem = DictMemoryInterface()
    device = FSAMMIO(params=params, memory=mem)

    check("initial_state", int.from_bytes(device.load(REG_STATE, 4), "little"), STATE_IDLE)

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

    dma_ld_q = encode_dma(
        func=DMA_LD_SRAM, sram_addr=3, sram_stride=1,
        mem_addr=Q_BASE, mem_stride=8, size=8, repeat=4,
    )
    mx_load_q = encode_matrix(
        func=MX_LOAD_STATIONARY, spad_addr=3, spad_stride=1,
        acc_addr=0, acc_stride=1,
    )
    dma_ld_k = encode_dma(
        func=DMA_LD_SRAM, sram_addr=7, sram_stride=1,
        mem_addr=K_BASE, mem_stride=8, size=8, repeat=4,
    )
    mx_score = encode_matrix(
        func=MX_ATTENTION_SCORE_COMPUTE, spad_addr=7, spad_stride=1,
        acc_addr=0, acc_stride=1,
    )
    dma_ld_v = encode_dma(
        func=DMA_LD_SRAM, sram_addr=11, sram_stride=1,
        mem_addr=V_BASE, mem_stride=8, size=8, repeat=4,
    )
    mx_value = encode_matrix(
        func=MX_ATTENTION_VALUE_COMPUTE, spad_addr=11, spad_stride=1,
        acc_addr=1, acc_stride=1, zero=True,
    )
    mx_lse_scale = encode_matrix(
        func=MX_ATTENTION_LSE_NORM_SCALE, spad_addr=0, spad_stride=1,
        acc_addr=0, acc_stride=1,
    )
    mx_lse_norm = encode_matrix(
        func=MX_ATTENTION_LSE_NORM, spad_addr=0, spad_stride=1,
        acc_addr=1, acc_stride=1,
    )
    dma_st_o = encode_dma(
        func=DMA_ST_SRAM, sram_addr=1, sram_stride=1,
        mem_addr=O_BASE, mem_stride=16, size=16, repeat=4, is_accum=True,
    )
    fence = encode_fence(matrix=True, dma=True, stop=True)

    all_words = (
        dma_ld_q + mx_load_q + dma_ld_k + mx_score +
        dma_ld_v + mx_value + mx_lse_scale + mx_lse_norm +
        dma_st_o + fence
    )

    expected_enq = 0
    for w in all_words:
        device.store(REG_INST_QUEUE, w.to_bytes(4, "little"))
        expected_enq += 1

    enq_val = int.from_bytes(device.load(REG_ENQ_INST_CNT, 4), "little")
    check("enq_cnt", enq_val, expected_enq)

    device.store(REG_SET_ACTIVE, (1).to_bytes(4, "little"))

    state_val = int.from_bytes(device.load(REG_STATE, 4), "little")
    check("state_done", state_val, STATE_DONE)

    deq_val = int.from_bytes(device.load(REG_DEQ_INST_CNT, 4), "little")
    check("deq_cnt", deq_val, 10)

    raw_inst = int.from_bytes(device.load(REG_PERF_RAW_INST, 4), "little")
    check("raw_inst", raw_inst, 10)

    mx_inst = int.from_bytes(device.load(REG_PERF_MX_INST, 4), "little")
    check("mx_inst", mx_inst, 5)

    dma_inst = int.from_bytes(device.load(REG_PERF_DMA_INST, 4), "little")
    check("dma_inst", dma_inst, 4)

    fence_inst = int.from_bytes(device.load(REG_PERF_FENCE, 4), "little")
    check("fence_inst", fence_inst, 1)

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
    check("max_error", max_err < 0.01, True)

    device.store(REG_SET_ACTIVE, (1).to_bytes(4, "little"))
    state_after_rerun = int.from_bytes(device.load(REG_STATE, 4), "little")
    check("state_after_rerun", state_after_rerun, STATE_ACTIVE)

    if errors:
        print("FAILURES:")
        for e in errors:
            print(e)
        raise AssertionError(f"{len(errors)} test(s) failed")
    print("test_integration: all passed")


if __name__ == "__main__":
    test_register_map()
    test_integration()
