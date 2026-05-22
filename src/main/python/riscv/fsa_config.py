from dataclasses import dataclass
import math


@dataclass
class FSAParams:
    saRows: int
    saCols: int
    spadRows: int
    accRows: int
    spadBanks: int = 2
    accBanks: int = 2
    instructionQueueEntries: int = 256
    mxInflight: int = 8
    dmaLoadInflight: int = 16
    dmaStoreInflight: int = 8
    nMemPorts: int = 1
    unitTestBuild: bool = False

    @property
    def spadAddrWidth(self) -> int:
        return max(1, (self.spadRows - 1).bit_length()) if self.spadRows > 0 else 1

    @property
    def accAddrWidth(self) -> int:
        return max(1, (self.accRows - 1).bit_length()) if self.accRows > 0 else 1

    @property
    def sramAddrWidth(self) -> int:
        return max(self.spadAddrWidth, self.accAddrWidth)

    @property
    def dmaMaxInflight(self) -> int:
        return max(self.dmaLoadInflight, self.dmaStoreInflight)


def _default_fsa_params(rows: int, cols: int, mem_ports: int) -> FSAParams:
    return FSAParams(
        saRows=rows,
        saCols=cols,
        spadRows=2 * cols + 4 * rows,
        accRows=1 + rows,
        nMemPorts=mem_ports,
    )


def fsa4x4() -> FSAParams:
    return _default_fsa_params(4, 4, 4)


def fsa8x8() -> FSAParams:
    return _default_fsa_params(8, 8, 4)


def fsa16x16() -> FSAParams:
    return _default_fsa_params(16, 16, 8)


def fsa32x32() -> FSAParams:
    return _default_fsa_params(32, 32, 8)


REG_INST_QUEUE = 0x00
REG_SET_ACTIVE = 0x04
REG_STATE = 0x08
REG_PERF_EXEC_TIME = 0x0C
REG_PERF_MX_BUBBLE = 0x10
REG_PERF_MX_ACTIVE = 0x14
REG_PERF_DMA_ACTIVE = 0x18
REG_PERF_RAW_INST = 0x1C
REG_PERF_MX_INST = 0x20
REG_PERF_DMA_INST = 0x24
REG_PERF_FENCE = 0x28
REG_ENQ_INST_CNT = 0x2C
REG_DEQ_INST_CNT = 0x30

FSA_MMIO_SIZE = 0x100

STATE_IDLE = 0
STATE_ACTIVE = 1
STATE_DONE = 2

INST_FENCE = 0
INST_MATRIX = 1
INST_DMA = 2

MX_LOAD_STATIONARY = 0
MX_ATTENTION_SCORE_COMPUTE = 1
MX_ATTENTION_VALUE_COMPUTE = 2
MX_ATTENTION_LSE_NORM_SCALE = 3
MX_ATTENTION_LSE_NORM = 4

DMA_LD_SRAM = 0
DMA_ST_SRAM = 1

I_TYPE_BITS = 3
N_SEMAPHORES = 32
SEM_ID_BITS = 5
SEM_VALUE_BITS = 3
MX_FUNC_BITS = 5
SPAD_MAX_ADDR_BITS = 20
SPAD_STRIDE_BITS = 5
ACC_MAX_ADDR_BITS = 20
ACC_STRIDE_BITS = 5
SRAM_MAX_ADDR_BITS = max(SPAD_MAX_ADDR_BITS, ACC_MAX_ADDR_BITS)
SRAM_STRIDE_BITS = max(SPAD_STRIDE_BITS, ACC_STRIDE_BITS)
DMA_FUNC_BITS = 4
DMA_SIZE_BITS = 10
DMA_REPEAT_BITS = 9
MEM_MAX_ADDR_BITS = 39
MEM_STRIDE_1_BITS = 6
MEM_STRIDE_2_BITS = 15
MEM_STRIDE_BITS = MEM_STRIDE_1_BITS + MEM_STRIDE_2_BITS

SPAD_CONST_ONE = 0
SPAD_CONST_ATTENTION_SCALE = 1
SPAD_CONST_EXP2_SLOPES = 2

ACC_CONST_ZERO = 0

ACC_CMD_EXP_S1 = 0
ACC_CMD_EXP_S2 = 1
ACC_CMD_ACC_SA = 2
ACC_CMD_ACC = 3
ACC_CMD_SET_SCALE = 4
ACC_CMD_RECIPROCAL = 5

CMP_CMD_UPDATE = 0
CMP_CMD_PROP_MAX = 1
CMP_CMD_PROP_MAX_DIFF = 2
CMP_CMD_PROP_ZERO = 3
CMP_CMD_RESET = 4
CMP_CMD_PROP_EXP2_INTERCEPTS = 5
