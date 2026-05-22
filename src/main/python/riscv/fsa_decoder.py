from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

from riscv.fsa_config import (
    I_TYPE_BITS,
    MX_FUNC_BITS,
    SPAD_MAX_ADDR_BITS,
    SPAD_STRIDE_BITS,
    ACC_MAX_ADDR_BITS,
    ACC_STRIDE_BITS,
    SRAM_MAX_ADDR_BITS,
    SRAM_STRIDE_BITS,
    DMA_FUNC_BITS,
    DMA_SIZE_BITS,
    DMA_REPEAT_BITS,
    MEM_MAX_ADDR_BITS,
    MEM_STRIDE_1_BITS,
    MEM_STRIDE_2_BITS,
    MEM_STRIDE_BITS,
    INST_FENCE,
    INST_MATRIX,
    INST_DMA,
    MX_LOAD_STATIONARY,
    DMA_LD_SRAM,
)


def bit_slice(value: int, hi: int, lo: int) -> int:
    mask = (1 << (hi - lo + 1)) - 1
    return (value >> lo) & mask


def sign_extend(value: int, bits: int) -> int:
    if value & (1 << (bits - 1)):
        return value - (1 << bits)
    return value


@dataclass(frozen=True)
class FenceInstruction:
    instType: int
    matrix: bool
    dma: bool
    stop: bool

    @staticmethod
    def decode(word: int) -> FenceInstruction:
        return FenceInstruction(
            instType=bit_slice(word, 2, 0),
            matrix=bool(bit_slice(word, 3, 3)),
            dma=bool(bit_slice(word, 4, 4)),
            stop=bool(bit_slice(word, 5, 5)),
        )


@dataclass(frozen=True)
class MatrixInstruction:
    instType: int
    semId: int
    acquireValid: bool
    acquireSemValue: int
    releaseValid: bool
    releaseSemValue: int
    func: int
    waitPrevAcc: bool
    spad_addr: int
    spad_stride: int
    revInput: bool
    revOutput: bool
    delayOutput: bool
    acc_addr: int
    acc_stride: int
    zero: bool

    @staticmethod
    def decode(word0: int, word1: int, word2: int) -> MatrixInstruction:
        header = word0
        spad = word1
        acc = word2
        return MatrixInstruction(
            instType=bit_slice(header, 2, 0),
            semId=bit_slice(header, 7, 3),
            acquireValid=bool(bit_slice(header, 8, 8)),
            acquireSemValue=bit_slice(header, 11, 9),
            releaseValid=bool(bit_slice(header, 12, 12)),
            releaseSemValue=bit_slice(header, 15, 13),
            func=bit_slice(header, 20, 16),
            waitPrevAcc=bool(bit_slice(header, 21, 21)),
            spad_addr=bit_slice(spad, SPAD_MAX_ADDR_BITS - 1, 0),
            spad_stride=sign_extend(
                bit_slice(spad, SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS - 1, SPAD_MAX_ADDR_BITS),
                SPAD_STRIDE_BITS,
            ),
            revInput=bool(bit_slice(spad, SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS, SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS)),
            revOutput=bool(bit_slice(spad, SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 1, SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 1)),
            delayOutput=bool(bit_slice(spad, SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 2, SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 2)),
            acc_addr=bit_slice(acc, ACC_MAX_ADDR_BITS - 1, 0),
            acc_stride=sign_extend(
                bit_slice(acc, ACC_MAX_ADDR_BITS + ACC_STRIDE_BITS - 1, ACC_MAX_ADDR_BITS),
                ACC_STRIDE_BITS,
            ),
            zero=bool(bit_slice(acc, ACC_MAX_ADDR_BITS + ACC_STRIDE_BITS, ACC_MAX_ADDR_BITS + ACC_STRIDE_BITS)),
        )


@dataclass(frozen=True)
class DMAInstruction:
    instType: int
    semId: int
    acquireValid: bool
    acquireSemValue: int
    releaseValid: bool
    releaseSemValue: int
    func: int
    repeat: int
    sram_addr: int
    sram_stride: int
    isAccum: bool
    mem_stride1: int
    mem_addr: int
    stride2: int
    size: int
    memStride: int

    @staticmethod
    def decode(word0: int, word1: int, word2: int, word3: int) -> DMAInstruction:
        header = word0
        sram = word1
        mem = word2 | (word3 << 32)
        mem_stride1 = bit_slice(
            sram,
            SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + MEM_STRIDE_1_BITS,
            SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1,
        )
        stride2 = bit_slice(mem, MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS - 1, MEM_MAX_ADDR_BITS)
        mem_stride_raw = (mem_stride1 << MEM_STRIDE_2_BITS) | stride2
        memStride = sign_extend(mem_stride_raw, MEM_STRIDE_BITS)
        return DMAInstruction(
            instType=bit_slice(header, 2, 0),
            semId=bit_slice(header, 7, 3),
            acquireValid=bool(bit_slice(header, 8, 8)),
            acquireSemValue=bit_slice(header, 11, 9),
            releaseValid=bool(bit_slice(header, 12, 12)),
            releaseSemValue=bit_slice(header, 15, 13),
            func=bit_slice(header, 16 + DMA_FUNC_BITS - 1, 16),
            repeat=bit_slice(header, 16 + DMA_FUNC_BITS + DMA_REPEAT_BITS - 1, 16 + DMA_FUNC_BITS),
            sram_addr=bit_slice(sram, SRAM_MAX_ADDR_BITS - 1, 0),
            sram_stride=sign_extend(
                bit_slice(sram, SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS - 1, SRAM_MAX_ADDR_BITS),
                SRAM_STRIDE_BITS,
            ),
            isAccum=bool(bit_slice(sram, SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS, SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS)),
            mem_stride1=mem_stride1,
            mem_addr=bit_slice(mem, MEM_MAX_ADDR_BITS - 1, 0),
            stride2=stride2,
            size=bit_slice(mem, MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS + DMA_SIZE_BITS - 1, MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS),
            memStride=memStride,
        )


class InstructionMerger:
    def __init__(self, n: int):
        self._n = n
        self._buf: List[int] = []
        self._cnt: int = 0

    def feed(self, word: int) -> Optional[int]:
        self._buf.append(word & 0xFFFFFFFF)
        self._cnt += 1
        if self._cnt == self._n:
            result = 0
            for i in range(self._n):
                result |= self._buf[i] << (32 * i)
            self._buf = []
            self._cnt = 0
            return result
        return None

    @property
    def inflight(self) -> bool:
        return self._cnt != 0

    def reset(self):
        self._buf = []
        self._cnt = 0


class Decoder:
    def __init__(self):
        self._mx_merger = InstructionMerger(3)
        self._dma_merger = InstructionMerger(4)
        self._sel: Optional[str] = None

    def feed(self, word: int) -> Optional[Union[FenceInstruction, MatrixInstruction, DMAInstruction]]:
        word &= 0xFFFFFFFF
        if self._sel is None:
            inst_type = bit_slice(word, I_TYPE_BITS - 1, 0)
            if inst_type == INST_FENCE:
                return FenceInstruction.decode(word)
            elif inst_type == INST_MATRIX:
                self._sel = "mx"
            elif inst_type == INST_DMA:
                self._sel = "dma"
            else:
                raise ValueError(f"Unknown instType: {inst_type}")

        if self._sel == "mx":
            result = self._mx_merger.feed(word)
            if result is not None:
                self._sel = None
                return MatrixInstruction.decode(
                    bit_slice(result, 31, 0),
                    bit_slice(result, 63, 32),
                    bit_slice(result, 95, 64),
                )
        elif self._sel == "dma":
            result = self._dma_merger.feed(word)
            if result is not None:
                self._sel = None
                return DMAInstruction.decode(
                    bit_slice(result, 31, 0),
                    bit_slice(result, 63, 32),
                    bit_slice(result, 95, 64),
                    bit_slice(result, 127, 96),
                )
        return None

    def reset(self):
        self._mx_merger.reset()
        self._dma_merger.reset()
        self._sel = None


def test_decoder():
    errors = []

    def check(name, actual, expected):
        if actual != expected:
            errors.append(f"  {name}: expected {expected}, got {actual}")

    check("bit_slice(0b10110, 4, 2)", bit_slice(0b10110, 4, 2), 0b101)
    check("sign_extend(0b11111, 5)", sign_extend(0b11111, 5), -1)
    check("sign_extend(0b01111, 5)", sign_extend(0b01111, 5), 15)

    fence_word = (1 << 5) | (1 << 4) | (1 << 3) | 0
    fence = FenceInstruction.decode(fence_word)
    check("fence.instType", fence.instType, 0)
    check("fence.matrix", fence.matrix, True)
    check("fence.dma", fence.dma, True)
    check("fence.stop", fence.stop, True)

    fence2 = FenceInstruction.decode(0)
    check("fence2.matrix", fence2.matrix, False)
    check("fence2.dma", fence2.dma, False)
    check("fence2.stop", fence2.stop, False)

    mx_header = (
        MX_LOAD_STATIONARY << 16
        | 1 << 12
        | 5 << 13
        | 1 << 8
        | 3 << 9
        | 17 << 3
        | INST_MATRIX
    )
    mx_header |= 1 << 21

    mx_spad = (
        0x3FFFF
        | (0b11111 << SPAD_MAX_ADDR_BITS)
        | (1 << (SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS))
        | (1 << (SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 2))
    )

    mx_acc = (
        0x1234
        | (0b11100 << ACC_MAX_ADDR_BITS)
        | (1 << (ACC_MAX_ADDR_BITS + ACC_STRIDE_BITS))
    )

    mx = MatrixInstruction.decode(mx_header, mx_spad, mx_acc)
    check("mx.instType", mx.instType, INST_MATRIX)
    check("mx.semId", mx.semId, 17)
    check("mx.acquireValid", mx.acquireValid, True)
    check("mx.acquireSemValue", mx.acquireSemValue, 3)
    check("mx.releaseValid", mx.releaseValid, True)
    check("mx.releaseSemValue", mx.releaseSemValue, 5)
    check("mx.func", mx.func, MX_LOAD_STATIONARY)
    check("mx.waitPrevAcc", mx.waitPrevAcc, True)
    check("mx.spad_addr", mx.spad_addr, 0x3FFFF)
    check("mx.spad_stride", mx.spad_stride, -1)
    check("mx.revInput", mx.revInput, True)
    check("mx.revOutput", mx.revOutput, False)
    check("mx.delayOutput", mx.delayOutput, True)
    check("mx.acc_addr", mx.acc_addr, 0x1234)
    check("mx.acc_stride", mx.acc_stride, -4)
    check("mx.zero", mx.zero, True)

    dma_header = (
        100 << 20
        | DMA_LD_SRAM << 16
        | 1 << 12
        | 6 << 13
        | 1 << 8
        | 2 << 9
        | 7 << 3
        | INST_DMA
    )

    dma_sram = (
        0xABCD
        | (0b11111 << SRAM_MAX_ADDR_BITS)
        | (1 << (SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS))
        | (0x2A << (SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1))
    )

    dma_mem_lo = 0xDEADBEEF
    dma_mem_hi = 0x12345678
    mem64 = dma_mem_lo | (dma_mem_hi << 32)
    expected_mem_addr = bit_slice(mem64, MEM_MAX_ADDR_BITS - 1, 0)
    expected_stride2 = bit_slice(mem64, MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS - 1, MEM_MAX_ADDR_BITS)
    expected_size = bit_slice(mem64, MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS + DMA_SIZE_BITS - 1, MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS)
    expected_mem_stride1 = bit_slice(dma_sram, SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + MEM_STRIDE_1_BITS, SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1)
    expected_memStride = sign_extend((expected_mem_stride1 << MEM_STRIDE_2_BITS) | expected_stride2, MEM_STRIDE_BITS)

    dma = DMAInstruction.decode(dma_header, dma_sram, dma_mem_lo, dma_mem_hi)
    check("dma.instType", dma.instType, INST_DMA)
    check("dma.semId", dma.semId, 7)
    check("dma.acquireValid", dma.acquireValid, True)
    check("dma.acquireSemValue", dma.acquireSemValue, 2)
    check("dma.releaseValid", dma.releaseValid, True)
    check("dma.releaseSemValue", dma.releaseSemValue, 6)
    check("dma.func", dma.func, DMA_LD_SRAM)
    check("dma.repeat", dma.repeat, 100)
    check("dma.sram_addr", dma.sram_addr, 0xABCD)
    check("dma.sram_stride", dma.sram_stride, -1)
    check("dma.isAccum", dma.isAccum, True)
    check("dma.mem_stride1", dma.mem_stride1, expected_mem_stride1)
    check("dma.mem_addr", dma.mem_addr, expected_mem_addr)
    check("dma.stride2", dma.stride2, expected_stride2)
    check("dma.size", dma.size, expected_size)
    check("dma.memStride", dma.memStride, expected_memStride)

    neg_ms1 = 0x3F
    neg_s2 = 0x7FFF
    neg_raw = (neg_ms1 << MEM_STRIDE_2_BITS) | neg_s2
    expected_neg = sign_extend(neg_raw, MEM_STRIDE_BITS)
    neg_sram = neg_ms1 << (SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1)
    neg_mem_lo = neg_s2 << MEM_MAX_ADDR_BITS
    neg_dma = DMAInstruction.decode(INST_DMA, neg_sram, neg_mem_lo, 0)
    check("dma.negMemStride", neg_dma.memStride, expected_neg)

    dec = Decoder()
    result = dec.feed(fence_word)
    check("decoder.fence_immediate", result is not None and isinstance(result, FenceInstruction), True)

    dec.reset()
    check("decoder.mx_word0", dec.feed(mx_header) is None, True)
    check("decoder.mx_word1", dec.feed(mx_spad) is None, True)
    result = dec.feed(mx_acc)
    check("decoder.mx_word2", result is not None and isinstance(result, MatrixInstruction), True)
    if result:
        check("decoder.mx.func", result.func, MX_LOAD_STATIONARY)
        check("decoder.mx.spad_stride", result.spad_stride, -1)

    dec.reset()
    check("decoder.dma_word0", dec.feed(dma_header) is None, True)
    check("decoder.dma_word1", dec.feed(dma_sram) is None, True)
    check("decoder.dma_word2", dec.feed(dma_mem_lo) is None, True)
    result = dec.feed(dma_mem_hi)
    check("decoder.dma_word3", result is not None and isinstance(result, DMAInstruction), True)
    if result:
        check("decoder.dma.func", result.func, DMA_LD_SRAM)
        check("decoder.dma.repeat", result.repeat, 100)

    neg_spad = 0b11111 << SPAD_MAX_ADDR_BITS
    neg_mx = MatrixInstruction.decode(INST_MATRIX, neg_spad, 0)
    check("mx.neg_stride", neg_mx.spad_stride, -1)

    pos_spad = 0b00010 << SPAD_MAX_ADDR_BITS
    pos_mx = MatrixInstruction.decode(INST_MATRIX, pos_spad, 0)
    check("mx.pos_stride", pos_mx.spad_stride, 2)

    if errors:
        print("FAILURES:")
        for e in errors:
            print(e)
        raise AssertionError(f"{len(errors)} test(s) failed")
    else:
        print("All tests passed!")


if __name__ == "__main__":
    test_decoder()
