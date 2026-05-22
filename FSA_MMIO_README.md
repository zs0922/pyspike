# FSA PySpike MMIO Device — Usage Guide

This module provides a **PySpike MMIO peripheral** that functionally models the FSA (Fusing FlashAttention within a Single Systolic Array) hardware accelerator. A RISC-V program running on Spike can drive the FSA through the **same MMIO register protocol** as real hardware, enabling software-only co-simulation of FlashAttention workloads.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   RISC-V Program                     │
│   (writes instructions via MMIO, polls state)        │
└──────────────┬──────────────────────────────────────┘
               │ store/load (AXI4 register interface)
               ▼
┌─────────────────────────────────────────────────────┐
│              FSAMMIO (PySpike Device)                │
│                                                      │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │  Decoder   │  │  FSAEngine│  │ Scratchpad/Acc  │ │
│  │ (32b words │→│ (NumPy    │→│ (SRAM models)    │ │
│  │  → insts)  │  │  compute) │  │                  │ │
│  └────────────┘  └──────────┘  └──────────────────┘ │
│                       ↕                              │
│              MemoryInterface                         │
│         (DictMemory / SimMemory)                     │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Standalone Mode (no PySpike installation required)

```python
import numpy as np
from riscv.fsa_config import fsa4x4
from riscv.fsa_mmio import FSAMMIO
from riscv.fsa_driver import FlashAttentionDriver
from riscv.fsa_sim_memory import DictMemoryInterface

# 1. Create device and driver
params = fsa4x4()
driver = FlashAttentionDriver(params)
device = FSAMMIO(params=params)

# 2. Prepare input data in memory
Q = np.random.randn(4, 4).astype(np.float16)
K = np.random.randn(4, 4).astype(np.float16)
V = np.random.randn(4, 4).astype(np.float16)

mem = device._memory  # DictMemoryInterface
Q_BASE, K_BASE, V_BASE, O_BASE = 0x1000, 0x2000, 0x3000, 0x4000
for i in range(4):
    mem.write(Q_BASE + i * 8, Q[i, :].tobytes())
    mem.write(K_BASE + i * 8, K[i, :].tobytes())
    mem.write(V_BASE + i * 8, V[i, :].tobytes())

# 3. Generate instruction sequence
#    spad_q=3 avoids constant rows at addresses 0-2
words = driver.flash_attention(
    q_addr=Q_BASE, k_addr=K_BASE, v_addr=V_BASE, o_addr=O_BASE,
    spad_q=3, spad_k=7, spad_v=11, acc_o=1, acc_lse=0,
)

# 4. Push instructions to the device
for w in words:
    device.store(0x00, w.to_bytes(4, 'little'))  # REG_INST_QUEUE

# 5. Trigger execution
device.store(0x04, (1).to_bytes(4, 'little'))    # REG_SET_ACTIVE

# 6. Read result
state = int.from_bytes(device.load(0x08, 4), 'little')  # REG_STATE
assert state == 2  # STATE_DONE

result = np.zeros((4, 4), dtype=np.float32)
for i in range(4):
    data = mem.read(O_BASE + i * 16, 16)
    result[i, :] = np.frombuffer(data, dtype=np.float32)

# 7. Verify against numpy reference
dk = 4
S = Q.astype(np.float32) @ K.astype(np.float32).T / np.sqrt(dk)
S_max = np.max(S, axis=1, keepdims=True)
P = np.exp(S - S_max) / np.sum(np.exp(S - S_max), axis=1, keepdims=True)
expected = P @ V.astype(np.float32)
print(f"Max error: {np.max(np.abs(result - expected)):.6f}")
```

### PySpike Mode (with Spike RISC-V simulator)

```bash
# Build PySpike first
cd pyspike && pip install -e '.[dev]'

# Run with the pyspike CLI
pyspike --extlib=fsa_mmio --device=fsa_mmio,0x8000 program.elf
```

This registers the FSA MMIO device at address `0x8000` in the simulated system. The RISC-V program can then access FSA registers via load/store instructions to `0x8000 + offset`.

### Engine-Only Mode (bypass MMIO, direct instruction execution)

```python
from riscv.fsa_config import fsa4x4
from riscv.fsa_engine import FSAEngine
from riscv.fsa_sim_memory import DictMemoryInterface
from riscv.fsa_decoder import DMAInstruction, MatrixInstruction, FenceInstruction

params = fsa4x4()
mem = DictMemoryInterface()
engine = FSAEngine(params, mem)

# Push instruction objects directly
engine.inst_queue.append(DMAInstruction(...))
engine.inst_queue.append(MatrixInstruction(...))
engine.inst_queue.append(FenceInstruction(instType=0, matrix=True, dma=True, stop=True))

engine.execute()
```

## Register Map (aligned with AXI4FSA.scala)

The FSA MMIO device occupies 256 bytes of address space. The base address is configurable (hardware default: `0x8000`).

| Offset | Name | Access | Description |
|--------|------|--------|-------------|
| `0x00` | REG_INST_QUEUE | W | Push a 32-bit instruction word into the queue |
| `0x04` | REG_SET_ACTIVE | W | Write 1 to activate execution; transitions idle→active or done→active (resets perf counters) |
| `0x08` | REG_STATE | R | Current state: 0=idle, 1=active, 2=done |
| `0x0C` | REG_PERF_EXEC_TIME | R | Performance: total execution cycles |
| `0x10` | REG_PERF_MX_BUBBLE | R | Performance: SA idle cycles while active |
| `0x14` | REG_PERF_MX_ACTIVE | R | Performance: SA active cycles |
| `0x18` | REG_PERF_DMA_ACTIVE | R | Performance: DMA active cycles |
| `0x1C` | REG_PERF_RAW_INST | R | Performance: total raw instructions dequeued |
| `0x20` | REG_PERF_MX_INST | R | Performance: matrix instructions executed |
| `0x24` | REG_PERF_DMA_INST | R | Performance: DMA instructions executed |
| `0x28` | REG_PERF_FENCE | R | Performance: fence instructions executed |
| `0x2C` | REG_ENQ_INST_CNT | R | Number of 32-bit words enqueued |
| `0x30` | REG_DEQ_INST_CNT | R | Number of instructions dequeued (executed) |

### State Machine

```
         set_active        fence(stop=1)
  IDLE ──────────► ACTIVE ──────────────► DONE
   ▲                                        │
   └──────────── set_active ────────────────┘
              (resets perf counters)
```

## Instruction Format

Instructions are pushed as 32-bit words to `REG_INST_QUEUE` (offset `0x00`). The lowest 3 bits of the first word determine the instruction type. The decoder collects the appropriate number of words before dispatching.

### Fence Instruction (1 word, instType=0)

```
Bits:  [2:0] instType = 0
       [3]   matrix     — wait for matrix engine
       [4]   dma        — wait for DMA engine
       [5]   stop       — transition to DONE state
```

### Matrix Instruction (3 words, instType=1)

**Word 0 — Header:**
```
Bits:  [2:0]   instType = 1
       [7:3]   semId
       [8]     acquireValid
       [11:9]  acquireSemValue
       [12]    releaseValid
       [15:13] releaseSemValue
       [20:16] func
       [21]    waitPrevAcc
```

**Word 1 — Scratchpad:**
```
Bits:  [19:0]  addr
       [24:20] stride (signed 5-bit)
       [25]    revInput
       [26]    revOutput
       [27]    delayOutput
```

**Word 2 — Accumulator:**
```
Bits:  [19:0]  addr
       [24:20] stride (signed 5-bit)
       [25]    zero
```

#### Matrix Functions

| Code | Name | Description |
|------|------|-------------|
| 0 | LOAD_STATIONARY | Load Q from scratchpad into systolic array registers |
| 1 | ATTENTION_SCORE_COMPUTE | Compute S = Q @ K^T / sqrt(dk) with online softmax |
| 2 | ATTENTION_VALUE_COMPUTE | Compute O = P @ V with accumulation |
| 3 | ATTENTION_LSE_NORM_SCALE | Compute 1/lse (reciprocal of log-sum-exp) |
| 4 | ATTENTION_LSE_NORM | Final normalization: O = O * (1/lse) |

### DMA Instruction (4 words, instType=2)

**Word 0 — Header:**
```
Bits:  [2:0]   instType = 2
       [7:3]   semId
       [8]     acquireValid
       [11:9]  acquireSemValue
       [12]    releaseValid
       [15:13] releaseSemValue
       [19:16] func
       [28:20] repeat (9-bit)
```

**Word 1 — SRAM:**
```
Bits:  [19:0]  addr
       [24:20] stride (signed 5-bit)
       [25]    isAccum (0=scratchpad, 1=accumulator)
       [31:26] mem_stride1 (upper 6 bits of memory stride)
```

**Words 2-3 — Memory (64-bit):**
```
Bits:  [38:0]  addr
       [53:39] stride2 (lower 15 bits of memory stride)
       [63:54] size (transfer size in bytes per repeat)
```

`memStride = sign_extend(mem_stride1 ## stride2)` — 21-bit signed value.

#### DMA Functions

| Code | Name | Description |
|------|------|-------------|
| 0 | LD_SRAM | Load from main memory to scratchpad/accumulator |
| 1 | ST_SRAM | Store from accumulator to main memory |

## Scratchpad and Accumulator Addressing

### Constant Rows (reserved, do not overwrite)

| Address | Scratchpad Constant | Accumulator Constant |
|---------|--------------------|--------------------|
| 0 | ONE (1.0 in each element) | ZERO (0.0 in each element) |
| 1 | AttentionScale (log2(e)/sqrt(dk)) | — |
| 2 | Exp2Slopes (piecewise-linear slopes) | — |

**Important**: User data must start at scratchpad row 3 or higher to avoid corrupting constants.

### Recommended Scratchpad Layout (4x4 config, spadRows=24)

| Rows | Content |
|------|---------|
| 0-2 | Constants (reserved) |
| 3-6 | Q matrix (4 rows) |
| 7-10 | K matrix (4 rows) |
| 11-14 | V matrix (4 rows) |
| 15-23 | Available for double-buffering |

### Accumulator Layout (4x4 config, accRows=5)

| Row | Content |
|-----|---------|
| 0 | LSE (log-sum-exp) |
| 1-4 | Output O (4 rows) |

## FlashAttention Execution Sequence

A complete FlashAttention inner-loop iteration follows this sequence:

```
1. DMA LD_SRAM     → Load Q from memory to scratchpad (release semaphore 0)
2. MX LOAD_STATIONARY → Load Q into SA registers (acquire sem 0, release sem 1)
3. DMA LD_SRAM     → Load K from memory to scratchpad (release sem 1)
4. MX ATTENTION_SCORE_COMPUTE → S = Q @ K^T / sqrt(dk) + online softmax (acquire sem 1, release sem 2)
5. DMA LD_SRAM     → Load V from memory to scratchpad (release sem 2)
6. MX ATTENTION_VALUE_COMPUTE → O = P @ V (acquire sem 2, release sem 3)
7. MX ATTENTION_LSE_NORM_SCALE → scale = 1/lse (release sem 3)
8. MX ATTENTION_LSE_NORM → O = O * scale (acquire sem 3)
9. DMA ST_SRAM     → Store O from accumulator to memory
10. FENCE(stop=1)  → Transition to DONE state
```

Semaphores coordinate DMA loads with matrix computation, enabling overlap.

## Using the FlashAttentionDriver

The `FlashAttentionDriver` class generates the complete instruction sequence with proper semaphore coordination:

```python
from riscv.fsa_config import fsa4x4
from riscv.fsa_driver import FlashAttentionDriver

driver = FlashAttentionDriver(fsa4x4())

# Generate all instruction words for one FlashAttention iteration
words = driver.flash_attention(
    q_addr=0x1000,          # Q matrix address in main memory
    k_addr=0x2000,          # K matrix address in main memory
    v_addr=0x3000,          # V matrix address in main memory
    o_addr=0x4000,          # Output address in main memory
    spad_q=3,               # Scratchpad start row for Q (must be >= 3)
    spad_k=7,               # Scratchpad start row for K
    spad_v=11,              # Scratchpad start row for V
    acc_o=1,                # Accumulator start row for O (row 0 = LSE)
    acc_lse=0,              # Accumulator row for LSE
)
```

### Individual Instruction Methods

```python
# DMA: Load Q (4 rows of 4 FP16 elements = 8 bytes/row)
words = driver.load_q(spad_addr=3, mem_addr=0x1000, rows=4, cols=4,
                      sem_id=0, release_sem=0)

# Matrix: Load stationary (Q into SA)
words = driver.load_stationary(spad_addr=3, sem_id=0,
                               acquire_sem=0, release_sem=1)

# DMA: Store O from accumulator
words = driver.store_o(acc_addr=1, mem_addr=0x4000, rows=4, cols=4)

# Fence with stop
words = driver.fence(stop=True, matrix=True, dma=True)
```

## Using the Instruction Encoder/Decoder

For low-level instruction construction:

```python
from riscv.fsa_encoder import encode_fence, encode_matrix, encode_dma
from riscv.fsa_decoder import Decoder

# Encode instructions to 32-bit words
fence_words = encode_fence(matrix=True, dma=True, stop=True)
mx_words = encode_matrix(func=0, spad_addr=3, spad_stride=0,
                          acc_addr=0, acc_stride=0)
dma_words = encode_dma(func=0, sram_addr=3, sram_stride=1,
                        mem_addr=0x1000, mem_stride=8, size=8, repeat=4)

# Decode 32-bit words back to instructions
decoder = Decoder()
for w in fence_words + mx_words + dma_words:
    result = decoder.feed(w)
    if result is not None:
        print(f"Decoded: {result}")
```

## Configuration Parameters

| Config | SA Size | Mem Ports | SpadRows | AccRows |
|--------|---------|-----------|----------|---------|
| `fsa4x4()` | 4×4 | 4 | 24 | 5 |
| `fsa8x8()` | 8×8 | 4 | 48 | 9 |
| `fsa16x16()` | 16×16 | 8 | 96 | 17 |
| `fsa32x32()` | 32×32 | 8 | 192 | 33 |

Custom configurations:

```python
from riscv.fsa_config import FSAParams

params = FSAParams(
    saRows=4, saCols=4,
    spadRows=24, accRows=5,
    nMemPorts=4,
    instructionQueueEntries=256,
)
```

## RISC-V Assembly Example

The FSA can be driven from RISC-V assembly using the hardware-aligned MMIO interface:

```asm
.equ FSA_MMIO_BASE,    0x8000
.equ REG_INST_QUEUE,   0x00
.equ REG_SET_ACTIVE,   0x04
.equ REG_STATE,        0x08

# Push instruction words (pre-computed via fsa_asm_encode.py)
la t0, FSA_MMIO_BASE
li t5, 0x00000009       # First instruction word
sw t5, REG_INST_QUEUE(t0)
# ... push remaining words ...

# Activate
li t4, 1
sw t4, REG_SET_ACTIVE(t0)

# Poll until done
poll:
lw t5, REG_STATE(t0)
li t6, 2                 # STATE_DONE
beq t5, t6, done
j poll

done:
# Read results from memory via DMA store
```

Use `fsa_asm_encode.py` to generate the `.equ` constants for instruction words:

```bash
cd pyspike/src/main/python/riscv/asm
python3 fsa_asm_encode.py
```

## Running Tests

```bash
# Module tests (from pyspike/src/main/python)
PYTHONPATH=. python3 -c "
from riscv.fsa_mmio import test_register_map, test_integration
from riscv.fsa_decoder import test_decoder
from riscv.fsa_memory import test_memory
from riscv.fsa_engine import test_engine
from riscv.fsa_encoder import test_encoder
from riscv.fsa_driver import test_driver
test_register_map(); test_decoder(); test_memory()
test_engine(); test_encoder(); test_driver(); test_integration()
"

# End-to-end tests
cd pyspike
PYTHONPATH=src/main/python python3 -c "
from tests.test_fsa_e2e import *
test_fsa_4x4_fp16(); test_fsa_8x8_fp16()
test_fsa_multiple_iterations(); test_perf_counters()
test_state_transitions(); test_fence_stop()
"

# Standalone demo
PYTHONPATH=src/main/python python3 examples/fsa_standalone_demo.py
```

## File Reference

| File | Description |
|------|-------------|
| `riscv/fsa_config.py` | FSAParams, register offsets, ISA constants |
| `riscv/fsa_decoder.py` | Instruction decoder (32-bit words → structured objects) |
| `riscv/fsa_encoder.py` | Instruction encoder (structured objects → 32-bit words) |
| `riscv/fsa_memory.py` | Scratchpad, Accumulator, Semaphores models |
| `riscv/fsa_engine.py` | Functional simulation engine (NumPy FlashAttention) |
| `riscv/fsa_driver.py` | High-level FlashAttentionDriver API |
| `riscv/fsa_mmio.py` | PySpike MMIO device (register interface + engine integration) |
| `riscv/fsa_sim_memory.py` | MemoryInterface ABC, DictMemoryInterface |
| `riscv/fsa_shmem.py` | **Deprecated** — Verilator shared-memory IPC (use fsa_engine.py) |
| `riscv/asm/fsa_asm.S` | RISC-V assembly test program |
| `riscv/asm/fsa_asm_encode.py` | Generates assembly instruction constants |
| `tests/test_fsa_e2e.py` | End-to-end test suite |
| `examples/fsa_standalone_demo.py` | Standalone demo (no PySpike required) |
| `examples/fsa_pyspike_demo.py` | PySpike integration demo |

## Limitations

- **SimMemoryInterface**: PySpike's `sim_t` does not currently expose memory read/write APIs through its Python bindings. DMA instructions in PySpike mode will fall back to `DictMemoryInterface`, meaning DMA data transfers operate on a disconnected memory space. This will be addressed in a future update when C++ bindings are extended.
- **Cycle accuracy**: The functional model is not cycle-accurate. It produces numerically correct results but does not model pipeline timing or resource contention.
- **Piecewise-linear exp2**: The hardware uses a PWL approximation for 2^x. The functional model uses `numpy.exp2()` for better accuracy, which may cause minor numerical differences vs. hardware.
