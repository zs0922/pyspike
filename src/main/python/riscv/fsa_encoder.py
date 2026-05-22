from __future__ import annotations

from riscv.fsa_config import (
    INST_FENCE,
    INST_MATRIX,
    INST_DMA,
    SPAD_MAX_ADDR_BITS,
    SPAD_STRIDE_BITS,
    ACC_MAX_ADDR_BITS,
    ACC_STRIDE_BITS,
    SRAM_MAX_ADDR_BITS,
    SRAM_STRIDE_BITS,
    MEM_MAX_ADDR_BITS,
    MEM_STRIDE_1_BITS,
    MEM_STRIDE_2_BITS,
    MEM_STRIDE_BITS,
    DMA_FUNC_BITS,
    DMA_REPEAT_BITS,
    DMA_SIZE_BITS,
)
from riscv.fsa_decoder import (
    Decoder,
    FenceInstruction,
    MatrixInstruction,
    DMAInstruction,
)


def to_unsigned(value: int, bits: int) -> int:
    if value < 0:
        return value + (1 << bits)
    return value


def encode_fence(matrix: bool, dma: bool, stop: bool) -> list[int]:
    word = INST_FENCE
    if matrix:
        word |= 1 << 3
    if dma:
        word |= 1 << 4
    if stop:
        word |= 1 << 5
    return [word & 0xFFFFFFFF]


def encode_matrix(
    func: int,
    spad_addr: int,
    spad_stride: int,
    acc_addr: int,
    acc_stride: int,
    rev_input: bool = False,
    rev_output: bool = False,
    delay_output: bool = False,
    zero: bool = False,
    wait_prev_acc: bool = False,
    sem_id: int = 0,
    acquire_valid: bool = False,
    acquire_sem_value: int = 0,
    release_valid: bool = False,
    release_sem_value: int = 0,
) -> list[int]:
    header = INST_MATRIX
    header |= (sem_id & ((1 << 5) - 1)) << 3
    if acquire_valid:
        header |= 1 << 8
    header |= (acquire_sem_value & ((1 << 3) - 1)) << 9
    if release_valid:
        header |= 1 << 12
    header |= (release_sem_value & ((1 << 3) - 1)) << 13
    header |= (func & ((1 << 5) - 1)) << 16
    if wait_prev_acc:
        header |= 1 << 21

    spad = spad_addr & ((1 << SPAD_MAX_ADDR_BITS) - 1)
    spad |= (to_unsigned(spad_stride, SPAD_STRIDE_BITS) & ((1 << SPAD_STRIDE_BITS) - 1)) << SPAD_MAX_ADDR_BITS
    if rev_input:
        spad |= 1 << (SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS)
    if rev_output:
        spad |= 1 << (SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 1)
    if delay_output:
        spad |= 1 << (SPAD_MAX_ADDR_BITS + SPAD_STRIDE_BITS + 2)

    acc = acc_addr & ((1 << ACC_MAX_ADDR_BITS) - 1)
    acc |= (to_unsigned(acc_stride, ACC_STRIDE_BITS) & ((1 << ACC_STRIDE_BITS) - 1)) << ACC_MAX_ADDR_BITS
    if zero:
        acc |= 1 << (ACC_MAX_ADDR_BITS + ACC_STRIDE_BITS)

    return [header & 0xFFFFFFFF, spad & 0xFFFFFFFF, acc & 0xFFFFFFFF]


def encode_dma(
    func: int,
    sram_addr: int,
    sram_stride: int,
    mem_addr: int,
    mem_stride: int,
    size: int,
    repeat: int,
    is_accum: bool = False,
    sem_id: int = 0,
    acquire_valid: bool = False,
    acquire_sem_value: int = 0,
    release_valid: bool = False,
    release_sem_value: int = 0,
) -> list[int]:
    header = INST_DMA
    header |= (sem_id & ((1 << 5) - 1)) << 3
    if acquire_valid:
        header |= 1 << 8
    header |= (acquire_sem_value & ((1 << 3) - 1)) << 9
    if release_valid:
        header |= 1 << 12
    header |= (release_sem_value & ((1 << 3) - 1)) << 13
    header |= (func & ((1 << DMA_FUNC_BITS) - 1)) << 16
    header |= (repeat & ((1 << DMA_REPEAT_BITS) - 1)) << (16 + DMA_FUNC_BITS)

    mem_stride_unsigned = to_unsigned(mem_stride, MEM_STRIDE_BITS)
    mem_stride1 = (mem_stride_unsigned >> MEM_STRIDE_2_BITS) & ((1 << MEM_STRIDE_1_BITS) - 1)
    stride2 = mem_stride_unsigned & ((1 << MEM_STRIDE_2_BITS) - 1)

    sram = sram_addr & ((1 << SRAM_MAX_ADDR_BITS) - 1)
    sram |= (to_unsigned(sram_stride, SRAM_STRIDE_BITS) & ((1 << SRAM_STRIDE_BITS) - 1)) << SRAM_MAX_ADDR_BITS
    if is_accum:
        sram |= 1 << (SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS)
    sram |= (mem_stride1 & ((1 << MEM_STRIDE_1_BITS) - 1)) << (SRAM_MAX_ADDR_BITS + SRAM_STRIDE_BITS + 1)

    mem = mem_addr & ((1 << MEM_MAX_ADDR_BITS) - 1)
    mem |= (stride2 & ((1 << MEM_STRIDE_2_BITS) - 1)) << MEM_MAX_ADDR_BITS
    mem |= (size & ((1 << DMA_SIZE_BITS) - 1)) << (MEM_MAX_ADDR_BITS + MEM_STRIDE_2_BITS)

    mem_lo = mem & 0xFFFFFFFF
    mem_hi = (mem >> 32) & 0xFFFFFFFF

    return [header & 0xFFFFFFFF, sram & 0xFFFFFFFF, mem_lo, mem_hi]


def test_encoder():
    errors = []

    def check(name, actual, expected):
        if actual != expected:
            errors.append(f"  {name}: expected {expected}, got {actual}")

    fence_words = encode_fence(matrix=True, dma=True, stop=True)
    check("fence_len", len(fence_words), 1)
    dec = Decoder()
    fence_inst = dec.feed(fence_words[0])
    check("fence_type", isinstance(fence_inst, FenceInstruction), True)
    if isinstance(fence_inst, FenceInstruction):
        check("fence.matrix", fence_inst.matrix, True)
        check("fence.dma", fence_inst.dma, True)
        check("fence.stop", fence_inst.stop, True)

    fence2_words = encode_fence(matrix=False, dma=False, stop=False)
    dec2 = Decoder()
    fence2_inst = dec2.feed(fence2_words[0])
    check("fence2.matrix", fence2_inst.matrix, False)
    check("fence2.dma", fence2_inst.dma, False)
    check("fence2.stop", fence2_inst.stop, False)

    mx_words = encode_matrix(
        func=0, spad_addr=0x3FFFF, spad_stride=-1, acc_addr=0x1234, acc_stride=-4,
        rev_input=True, rev_output=False, delay_output=True, zero=True,
        wait_prev_acc=True, sem_id=17, acquire_valid=True, acquire_sem_value=3,
        release_valid=True, release_sem_value=5,
    )
    check("mx_len", len(mx_words), 3)
    dec3 = Decoder()
    check("mx_w0", dec3.feed(mx_words[0]) is None, True)
    check("mx_w1", dec3.feed(mx_words[1]) is None, True)
    mx_inst = dec3.feed(mx_words[2])
    check("mx_type", isinstance(mx_inst, MatrixInstruction), True)
    if isinstance(mx_inst, MatrixInstruction):
        check("mx.instType", mx_inst.instType, INST_MATRIX)
        check("mx.semId", mx_inst.semId, 17)
        check("mx.acquireValid", mx_inst.acquireValid, True)
        check("mx.acquireSemValue", mx_inst.acquireSemValue, 3)
        check("mx.releaseValid", mx_inst.releaseValid, True)
        check("mx.releaseSemValue", mx_inst.releaseSemValue, 5)
        check("mx.func", mx_inst.func, 0)
        check("mx.waitPrevAcc", mx_inst.waitPrevAcc, True)
        check("mx.spad_addr", mx_inst.spad_addr, 0x3FFFF)
        check("mx.spad_stride", mx_inst.spad_stride, -1)
        check("mx.revInput", mx_inst.revInput, True)
        check("mx.revOutput", mx_inst.revOutput, False)
        check("mx.delayOutput", mx_inst.delayOutput, True)
        check("mx.acc_addr", mx_inst.acc_addr, 0x1234)
        check("mx.acc_stride", mx_inst.acc_stride, -4)
        check("mx.zero", mx_inst.zero, True)

    mx_pos_words = encode_matrix(
        func=1, spad_addr=0, spad_stride=2, acc_addr=0, acc_stride=3,
    )
    dec_pos = Decoder()
    dec_pos.feed(mx_pos_words[0])
    dec_pos.feed(mx_pos_words[1])
    mx_pos = dec_pos.feed(mx_pos_words[2])
    check("mx_pos.spad_stride", mx_pos.spad_stride, 2)
    check("mx_pos.acc_stride", mx_pos.acc_stride, 3)

    dma_words = encode_dma(
        func=0, sram_addr=0xABCD, sram_stride=-1, mem_addr=0xDEADBEEF,
        mem_stride=0, size=256, repeat=100, is_accum=True,
        sem_id=7, acquire_valid=True, acquire_sem_value=2,
        release_valid=True, release_sem_value=6,
    )
    check("dma_len", len(dma_words), 4)
    dec4 = Decoder()
    check("dma_w0", dec4.feed(dma_words[0]) is None, True)
    check("dma_w1", dec4.feed(dma_words[1]) is None, True)
    check("dma_w2", dec4.feed(dma_words[2]) is None, True)
    dma_inst = dec4.feed(dma_words[3])
    check("dma_type", isinstance(dma_inst, DMAInstruction), True)
    if isinstance(dma_inst, DMAInstruction):
        check("dma.instType", dma_inst.instType, INST_DMA)
        check("dma.semId", dma_inst.semId, 7)
        check("dma.acquireValid", dma_inst.acquireValid, True)
        check("dma.acquireSemValue", dma_inst.acquireSemValue, 2)
        check("dma.releaseValid", dma_inst.releaseValid, True)
        check("dma.releaseSemValue", dma_inst.releaseSemValue, 6)
        check("dma.func", dma_inst.func, 0)
        check("dma.repeat", dma_inst.repeat, 100)
        check("dma.sram_addr", dma_inst.sram_addr, 0xABCD)
        check("dma.sram_stride", dma_inst.sram_stride, -1)
        check("dma.isAccum", dma_inst.isAccum, True)
        check("dma.mem_addr", dma_inst.mem_addr, 0xDEADBEEF & ((1 << MEM_MAX_ADDR_BITS) - 1))
        check("dma.memStride", dma_inst.memStride, 0)
        check("dma.size", dma_inst.size, 256)

    dma_neg_words = encode_dma(
        func=0, sram_addr=0, sram_stride=1, mem_addr=0,
        mem_stride=-1, size=8, repeat=1,
    )
    dec5 = Decoder()
    dec5.feed(dma_neg_words[0])
    dec5.feed(dma_neg_words[1])
    dec5.feed(dma_neg_words[2])
    dma_neg = dec5.feed(dma_neg_words[3])
    check("dma_neg.memStride", dma_neg.memStride, -1)

    check("to_unsigned_pos", to_unsigned(5, 5), 5)
    check("to_unsigned_neg", to_unsigned(-1, 5), 31)
    check("to_unsigned_neg2", to_unsigned(-4, 5), 28)

    if errors:
        print("FAILURES:")
        for e in errors:
            print(e)
        raise AssertionError(f"{len(errors)} test(s) failed")
    print("test_encoder: all passed")


if __name__ == "__main__":
    test_encoder()
