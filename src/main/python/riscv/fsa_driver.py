from __future__ import annotations

from riscv.fsa_config import (
    FSAParams,
    MX_LOAD_STATIONARY,
    MX_ATTENTION_SCORE_COMPUTE,
    MX_ATTENTION_VALUE_COMPUTE,
    MX_ATTENTION_LSE_NORM_SCALE,
    MX_ATTENTION_LSE_NORM,
    DMA_LD_SRAM,
    DMA_ST_SRAM,
    REG_INST_QUEUE,
    REG_SET_ACTIVE,
    REG_STATE,
    STATE_IDLE,
    STATE_ACTIVE,
    STATE_DONE,
)
from riscv.fsa_encoder import encode_dma, encode_matrix, encode_fence
from riscv.fsa_decoder import Decoder, FenceInstruction, MatrixInstruction, DMAInstruction


class FlashAttentionDriver:
    def __init__(self, params: FSAParams, mmio_base: int = 0x8000):
        self.params = params
        self.mmio_base = mmio_base
        self.elemWidth_bits = 16
        self.elemWidth_bytes = self.elemWidth_bits // 8
        self.accElemWidth_bits = 32
        self.accElemWidth_bytes = self.accElemWidth_bits // 8

    def _sem_flags(self, sem_id: int = 0, acquire_sem: int = -1, release_sem: int = 0):
        acquire_valid = acquire_sem >= 0
        acquire_sem_value = acquire_sem if acquire_valid else 0
        release_valid = release_sem >= 0
        release_sem_value = release_sem if release_valid else 0
        return dict(
            sem_id=sem_id,
            acquire_valid=acquire_valid,
            acquire_sem_value=acquire_sem_value,
            release_valid=release_valid,
            release_sem_value=release_sem_value,
        )

    def load_q(self, spad_addr: int, mem_addr: int, rows: int, cols: int,
               sem_id: int = 0, acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        size = cols * self.elemWidth_bytes
        return encode_dma(
            func=DMA_LD_SRAM,
            sram_addr=spad_addr,
            sram_stride=1,
            mem_addr=mem_addr,
            mem_stride=cols * self.elemWidth_bytes,
            size=size,
            repeat=rows,
            is_accum=False,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def load_k(self, spad_addr: int, mem_addr: int, rows: int, cols: int,
               sem_id: int = 0, acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        size = cols * self.elemWidth_bytes
        return encode_dma(
            func=DMA_LD_SRAM,
            sram_addr=spad_addr,
            sram_stride=1,
            mem_addr=mem_addr,
            mem_stride=cols * self.elemWidth_bytes,
            size=size,
            repeat=rows,
            is_accum=False,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def load_v(self, spad_addr: int, mem_addr: int, rows: int, cols: int,
               sem_id: int = 0, acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        size = cols * self.elemWidth_bytes
        return encode_dma(
            func=DMA_LD_SRAM,
            sram_addr=spad_addr,
            sram_stride=1,
            mem_addr=mem_addr,
            mem_stride=cols * self.elemWidth_bytes,
            size=size,
            repeat=rows,
            is_accum=False,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def load_stationary(self, spad_addr: int, sem_id: int = 0,
                        acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        return encode_matrix(
            func=MX_LOAD_STATIONARY,
            spad_addr=spad_addr,
            spad_stride=0,
            acc_addr=0,
            acc_stride=0,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def attention_score_compute(self, spad_addr: int, acc_addr: int, sem_id: int = 0,
                                acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        return encode_matrix(
            func=MX_ATTENTION_SCORE_COMPUTE,
            spad_addr=spad_addr,
            spad_stride=0,
            acc_addr=acc_addr,
            acc_stride=0,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def attention_value_compute(self, spad_addr: int, acc_addr: int, zero: bool = True,
                                sem_id: int = 0, acquire_sem: int = -1,
                                release_sem: int = 0) -> list[int]:
        return encode_matrix(
            func=MX_ATTENTION_VALUE_COMPUTE,
            spad_addr=spad_addr,
            spad_stride=0,
            acc_addr=acc_addr,
            acc_stride=0,
            zero=zero,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def attention_lse_norm_scale(self, acc_addr: int, sem_id: int = 0,
                                 acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        return encode_matrix(
            func=MX_ATTENTION_LSE_NORM_SCALE,
            spad_addr=0,
            spad_stride=0,
            acc_addr=acc_addr,
            acc_stride=0,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def attention_lse_norm(self, acc_addr: int, sem_id: int = 0,
                           acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        return encode_matrix(
            func=MX_ATTENTION_LSE_NORM,
            spad_addr=0,
            spad_stride=0,
            acc_addr=acc_addr,
            acc_stride=0,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def store_o(self, acc_addr: int, mem_addr: int, rows: int, cols: int,
                sem_id: int = 0, acquire_sem: int = -1, release_sem: int = 0) -> list[int]:
        size = cols * self.accElemWidth_bytes
        return encode_dma(
            func=DMA_ST_SRAM,
            sram_addr=acc_addr,
            sram_stride=1,
            mem_addr=mem_addr,
            mem_stride=cols * self.accElemWidth_bytes,
            size=size,
            repeat=rows,
            is_accum=True,
            **self._sem_flags(sem_id, acquire_sem, release_sem),
        )

    def fence(self, stop: bool = False, matrix: bool = True, dma: bool = True) -> list[int]:
        return encode_fence(matrix=matrix, dma=dma, stop=stop)

    def flash_attention(self, q_addr: int, k_addr: int, v_addr: int, o_addr: int,
                        spad_q: int, spad_k: int, spad_v: int,
                        acc_o: int, acc_lse: int) -> list[int]:
        rows = self.params.saRows
        cols = self.params.saCols
        words: list[int] = []

        words += self.load_q(spad_q, q_addr, rows, cols,
                             sem_id=0, acquire_sem=-1, release_sem=0)
        words += self.load_stationary(spad_q,
                                      sem_id=0, acquire_sem=0, release_sem=1)
        words += self.load_k(spad_k, k_addr, rows, cols,
                             sem_id=1, acquire_sem=-1, release_sem=1)
        words += self.attention_score_compute(spad_k, acc_lse,
                                              sem_id=1, acquire_sem=1, release_sem=2)
        words += self.load_v(spad_v, v_addr, rows, cols,
                             sem_id=2, acquire_sem=-1, release_sem=2)
        words += self.attention_value_compute(spad_v, acc_o, zero=True,
                                              sem_id=2, acquire_sem=2, release_sem=3)
        words += self.attention_lse_norm_scale(acc_lse,
                                               sem_id=3, acquire_sem=-1, release_sem=3)
        words += self.attention_lse_norm(acc_o,
                                         sem_id=3, acquire_sem=3, release_sem=-1)
        words += self.store_o(acc_o, o_addr, rows, cols,
                              sem_id=0, acquire_sem=-1, release_sem=-1)
        words += self.fence(stop=True, matrix=True, dma=True)

        return words


def test_driver():
    errors = []

    def check(name, actual, expected):
        if actual != expected:
            errors.append(f"  {name}: expected {expected}, got {actual}")

    params = FSAParams(saRows=4, saCols=4, spadRows=24, accRows=5)
    driver = FlashAttentionDriver(params, mmio_base=0x8000)

    check("elemWidth_bytes", driver.elemWidth_bytes, 2)
    check("accElemWidth_bytes", driver.accElemWidth_bytes, 4)

    ld_q = driver.load_q(spad_addr=0, mem_addr=0x80000000, rows=4, cols=4,
                         sem_id=0, acquire_sem=-1, release_sem=0)
    check("load_q_len", len(ld_q), 4)
    dec = Decoder()
    dec.feed(ld_q[0]); dec.feed(ld_q[1]); dec.feed(ld_q[2])
    inst = dec.feed(ld_q[3])
    check("load_q_type", isinstance(inst, DMAInstruction), True)
    if isinstance(inst, DMAInstruction):
        check("load_q_func", inst.func, DMA_LD_SRAM)
        check("load_q_sram_addr", inst.sram_addr, 0)
        check("load_q_sram_stride", inst.sram_stride, 1)
        check("load_q_repeat", inst.repeat, 4)
        check("load_q_size", inst.size, 8)
        check("load_q_isAccum", inst.isAccum, False)
        check("load_q_semId", inst.semId, 0)
        check("load_q_releaseValid", inst.releaseValid, True)
        check("load_q_releaseSemValue", inst.releaseSemValue, 0)
        check("load_q_acquireValid", inst.acquireValid, False)

    ld_k = driver.load_k(spad_addr=4, mem_addr=0x80000080, rows=4, cols=4,
                         sem_id=1, acquire_sem=-1, release_sem=1)
    check("load_k_len", len(ld_k), 4)
    dec2 = Decoder()
    dec2.feed(ld_k[0]); dec2.feed(ld_k[1]); dec2.feed(ld_k[2])
    inst2 = dec2.feed(ld_k[3])
    check("load_k_type", isinstance(inst2, DMAInstruction), True)
    if isinstance(inst2, DMAInstruction):
        check("load_k_func", inst2.func, DMA_LD_SRAM)
        check("load_k_sram_addr", inst2.sram_addr, 4)
        check("load_k_semId", inst2.semId, 1)
        check("load_k_releaseValid", inst2.releaseValid, True)
        check("load_k_releaseSemValue", inst2.releaseSemValue, 1)

    ld_v = driver.load_v(spad_addr=8, mem_addr=0x80000100, rows=4, cols=4,
                         sem_id=2, acquire_sem=-1, release_sem=2)
    check("load_v_len", len(ld_v), 4)
    dec3 = Decoder()
    dec3.feed(ld_v[0]); dec3.feed(ld_v[1]); dec3.feed(ld_v[2])
    inst3 = dec3.feed(ld_v[3])
    check("load_v_type", isinstance(inst3, DMAInstruction), True)
    if isinstance(inst3, DMAInstruction):
        check("load_v_func", inst3.func, DMA_LD_SRAM)
        check("load_v_sram_addr", inst3.sram_addr, 8)
        check("load_v_semId", inst3.semId, 2)

    ls = driver.load_stationary(spad_addr=0, sem_id=0, acquire_sem=0, release_sem=1)
    check("load_stationary_len", len(ls), 3)
    dec4 = Decoder()
    dec4.feed(ls[0]); dec4.feed(ls[1])
    inst4 = dec4.feed(ls[2])
    check("ls_type", isinstance(inst4, MatrixInstruction), True)
    if isinstance(inst4, MatrixInstruction):
        check("ls_func", inst4.func, MX_LOAD_STATIONARY)
        check("ls_spad_addr", inst4.spad_addr, 0)
        check("ls_semId", inst4.semId, 0)
        check("ls_acquireValid", inst4.acquireValid, True)
        check("ls_acquireSemValue", inst4.acquireSemValue, 0)
        check("ls_releaseValid", inst4.releaseValid, True)
        check("ls_releaseSemValue", inst4.releaseSemValue, 1)

    asc = driver.attention_score_compute(spad_addr=4, acc_addr=0, sem_id=1,
                                         acquire_sem=1, release_sem=2)
    check("asc_len", len(asc), 3)
    dec5 = Decoder()
    dec5.feed(asc[0]); dec5.feed(asc[1])
    inst5 = dec5.feed(asc[2])
    check("asc_type", isinstance(inst5, MatrixInstruction), True)
    if isinstance(inst5, MatrixInstruction):
        check("asc_func", inst5.func, MX_ATTENTION_SCORE_COMPUTE)
        check("asc_spad_addr", inst5.spad_addr, 4)
        check("asc_acc_addr", inst5.acc_addr, 0)
        check("asc_semId", inst5.semId, 1)
        check("asc_acquireValid", inst5.acquireValid, True)
        check("asc_acquireSemValue", inst5.acquireSemValue, 1)
        check("asc_releaseValid", inst5.releaseValid, True)
        check("asc_releaseSemValue", inst5.releaseSemValue, 2)

    avc = driver.attention_value_compute(spad_addr=8, acc_addr=1, zero=True,
                                         sem_id=2, acquire_sem=2, release_sem=3)
    check("avc_len", len(avc), 3)
    dec6 = Decoder()
    dec6.feed(avc[0]); dec6.feed(avc[1])
    inst6 = dec6.feed(avc[2])
    check("avc_type", isinstance(inst6, MatrixInstruction), True)
    if isinstance(inst6, MatrixInstruction):
        check("avc_func", inst6.func, MX_ATTENTION_VALUE_COMPUTE)
        check("avc_zero", inst6.zero, True)
        check("avc_semId", inst6.semId, 2)
        check("avc_acquireValid", inst6.acquireValid, True)
        check("avc_acquireSemValue", inst6.acquireSemValue, 2)

    lse_scale = driver.attention_lse_norm_scale(acc_addr=0, sem_id=3,
                                                 acquire_sem=-1, release_sem=3)
    check("lse_scale_len", len(lse_scale), 3)
    dec7 = Decoder()
    dec7.feed(lse_scale[0]); dec7.feed(lse_scale[1])
    inst7 = dec7.feed(lse_scale[2])
    check("lse_scale_type", isinstance(inst7, MatrixInstruction), True)
    if isinstance(inst7, MatrixInstruction):
        check("lse_scale_func", inst7.func, MX_ATTENTION_LSE_NORM_SCALE)
        check("lse_scale_acc_addr", inst7.acc_addr, 0)
        check("lse_scale_semId", inst7.semId, 3)
        check("lse_scale_releaseValid", inst7.releaseValid, True)
        check("lse_scale_releaseSemValue", inst7.releaseSemValue, 3)

    lse_norm = driver.attention_lse_norm(acc_addr=1, sem_id=3,
                                         acquire_sem=3, release_sem=-1)
    check("lse_norm_len", len(lse_norm), 3)
    dec8 = Decoder()
    dec8.feed(lse_norm[0]); dec8.feed(lse_norm[1])
    inst8 = dec8.feed(lse_norm[2])
    check("lse_norm_type", isinstance(inst8, MatrixInstruction), True)
    if isinstance(inst8, MatrixInstruction):
        check("lse_norm_func", inst8.func, MX_ATTENTION_LSE_NORM)
        check("lse_norm_acc_addr", inst8.acc_addr, 1)
        check("lse_norm_semId", inst8.semId, 3)
        check("lse_norm_acquireValid", inst8.acquireValid, True)
        check("lse_norm_acquireSemValue", inst8.acquireSemValue, 3)
        check("lse_norm_releaseValid", inst8.releaseValid, False)

    st_o = driver.store_o(acc_addr=1, mem_addr=0x80000200, rows=4, cols=4,
                          sem_id=0, acquire_sem=-1, release_sem=-1)
    check("store_o_len", len(st_o), 4)
    dec9 = Decoder()
    dec9.feed(st_o[0]); dec9.feed(st_o[1]); dec9.feed(st_o[2])
    inst9 = dec9.feed(st_o[3])
    check("store_o_type", isinstance(inst9, DMAInstruction), True)
    if isinstance(inst9, DMAInstruction):
        check("store_o_func", inst9.func, DMA_ST_SRAM)
        check("store_o_sram_addr", inst9.sram_addr, 1)
        check("store_o_isAccum", inst9.isAccum, True)
        check("store_o_repeat", inst9.repeat, 4)
        check("store_o_size", inst9.size, 16)
        check("store_o_acquireValid", inst9.acquireValid, False)
        check("store_o_releaseValid", inst9.releaseValid, False)

    fence_words = driver.fence(stop=True, matrix=True, dma=True)
    check("fence_len", len(fence_words), 1)
    dec10 = Decoder()
    inst10 = dec10.feed(fence_words[0])
    check("fence_type", isinstance(inst10, FenceInstruction), True)
    if isinstance(inst10, FenceInstruction):
        check("fence_stop", inst10.stop, True)
        check("fence_matrix", inst10.matrix, True)
        check("fence_dma", inst10.dma, True)

    full = driver.flash_attention(
        q_addr=0x80000000, k_addr=0x80000080, v_addr=0x80000100, o_addr=0x80000200,
        spad_q=3, spad_k=7, spad_v=11, acc_o=1, acc_lse=0,
    )
    expected_words = 4 + 3 + 4 + 3 + 4 + 3 + 3 + 3 + 4 + 1
    check("full_len", len(full), expected_words)

    dec_full = Decoder()
    decoded = []
    for w in full:
        result = dec_full.feed(w)
        if result is not None:
            decoded.append(result)
    expected_insts = 10
    check("full_decoded_count", len(decoded), expected_insts)

    check("decoded_0_dma_ldq", isinstance(decoded[0], DMAInstruction), True)
    check("decoded_1_mx_ls", isinstance(decoded[1], MatrixInstruction), True)
    if isinstance(decoded[1], MatrixInstruction):
        check("decoded_1_func", decoded[1].func, MX_LOAD_STATIONARY)
    check("decoded_2_dma_ldk", isinstance(decoded[2], DMAInstruction), True)
    check("decoded_3_mx_asc", isinstance(decoded[3], MatrixInstruction), True)
    if isinstance(decoded[3], MatrixInstruction):
        check("decoded_3_func", decoded[3].func, MX_ATTENTION_SCORE_COMPUTE)
    check("decoded_4_dma_ldv", isinstance(decoded[4], DMAInstruction), True)
    check("decoded_5_mx_avc", isinstance(decoded[5], MatrixInstruction), True)
    if isinstance(decoded[5], MatrixInstruction):
        check("decoded_5_func", decoded[5].func, MX_ATTENTION_VALUE_COMPUTE)
    check("decoded_6_mx_lse_scale", isinstance(decoded[6], MatrixInstruction), True)
    if isinstance(decoded[6], MatrixInstruction):
        check("decoded_6_func", decoded[6].func, MX_ATTENTION_LSE_NORM_SCALE)
    check("decoded_7_mx_lse_norm", isinstance(decoded[7], MatrixInstruction), True)
    if isinstance(decoded[7], MatrixInstruction):
        check("decoded_7_func", decoded[7].func, MX_ATTENTION_LSE_NORM)
    check("decoded_8_dma_sto", isinstance(decoded[8], DMAInstruction), True)
    if isinstance(decoded[8], DMAInstruction):
        check("decoded_8_func", decoded[8].func, DMA_ST_SRAM)
    check("decoded_9_fence", isinstance(decoded[9], FenceInstruction), True)
    if isinstance(decoded[9], FenceInstruction):
        check("decoded_9_stop", decoded[9].stop, True)

    if errors:
        print("FAILURES:")
        for e in errors:
            print(e)
        raise AssertionError(f"{len(errors)} test(s) failed")
    print("test_driver: all passed")


if __name__ == "__main__":
    test_driver()
