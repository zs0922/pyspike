# Plan: Align PySpike FSA MMIO Model to AXI4FSA Hardware (Plan 1)

## Overview

Rewrite the PySpike FSA MMIO device (`fsa_mmio.py`) to faithfully model the AXI4FSA hardware
register interface, implement a Python instruction decoder, and provide a NumPy-based functional
simulation of FlashAttention. This enables software-only co-simulation where a RISC-V program
running on Spike can drive the FSA accelerator through the same MMIO register protocol as real
hardware.

### Key Alignment Points

| Aspect | Current `fsa_mmio.py` | Target (matches AXI4FSA) |
|--------|----------------------|--------------------------|
| Register map | Custom (CTRL/STATUS/DATA_ADDR/...) | Hardware: inst_queue/set_active/state/perf_counters |
| Programming model | Register-driven (write Q/K/V directly) | Instruction-driven (push 32-bit instruction words) |
| Data movement | MMIO DATA register | DMA instructions (read/write main memory via sim_t) |
| State machine | IDLE→INIT→RUNNING→DONE | idle(0)→active(1)→done(2) |
| Address space | 0x10000 (64KB) | 0x100 (256 bytes, AddressSet(0x8000, 0xff)) |

### Hardware Register Map (AXI4FSA.scala:80-94)

| Offset | Type | Description |
|--------|------|-------------|
| 0x00 | W | Instruction queue enqueue (32-bit) |
| 0x04 | W | Set active (write 1 to activate) |
| 0x08 | R | State (0=idle, 1=active, 2=done) |
| 0x0C | R | Perf: execTime |
| 0x10 | R | Perf: mxBubble |
| 0x14 | R | Perf: mxActive |
| 0x18 | R | Perf: dmaActive |
| 0x1C | R | Perf: rawInst |
| 0x20 | R | Perf: mxInst |
| 0x24 | R | Perf: dmaInst |
| 0x28 | R | Perf: fence |
| 0x2C | R | Perf: enqInstCnt |
| 0x30 | R | Perf: deqInstCnt |

### Instruction Encoding (from ISA.scala)

- **Fence** (instType=0): 1 word (4 bytes)
- **Matrix** (instType=1): 3 words (12 bytes) — header + spad + acc
- **DMA** (instType=2): 4 words (16 bytes) — header + sram + mem(8B)

### Matrix Functions (MxFunc)

| Code | Name | ExecutionPlan |
|------|------|---------------|
| 0 | LOAD_STATIONARY | Load Q into SA registers |
| 1 | ATTENTION_SCORE_COMPUTE | S=Q@K, row-max, softmax, exp-sum |
| 2 | ATTENTION_VALUE_COMPUTE | O=P@V with accumulation |
| 3 | ATTENTION_LSE_NORM_SCALE | Compute 1/lse |
| 4 | ATTENTION_LSE_NORM | Final O normalization |

### DMA Functions

| Code | Name | Description |
|------|------|-------------|
| 0 | LD_SRAM | Load from memory to scratchpad |
| 1 | ST_SRAM | Store from accumulator to memory |

---

## Phase 1: FSA Configuration & Register Map Module

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 1 of the PySpike FSA MMIO alignment project.

**Goal**: Create the foundational FSA configuration module and rewrite the MMIO register map
to match the AXI4FSA hardware interface.

**Tasks**:

1. Create `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_config.py`:
   - Define FSAParams dataclass matching the hardware FSAParams (saRows, saCols, spadRows,
     accRows, nMemPorts, instructionQueueEntries, etc.)
   - Provide factory functions for standard configs: fsa4x4, fsa8x8, fsa16x16, fsa32x32
   - Define register offset constants matching AXI4FSA.scala:80-94:
     - REG_INST_QUEUE = 0x00 (write-only)
     - REG_SET_ACTIVE = 0x04 (write-only)
     - REG_STATE = 0x08 (read-only)
     - REG_PERF_EXEC_TIME = 0x0C (read-only)
     - REG_PERF_MX_BUBBLE = 0x10 (read-only)
     - REG_PERF_MX_ACTIVE = 0x14 (read-only)
     - REG_PERF_DMA_ACTIVE = 0x18 (read-only)
     - REG_PERF_RAW_INST = 0x1C (read-only)
     - REG_PERF_MX_INST = 0x20 (read-only)
     - REG_PERF_DMA_INST = 0x24 (read-only)
     - REG_PERF_FENCE = 0x28 (read-only)
     - REG_ENQ_INST_CNT = 0x2C (read-only)
     - REG_DEQ_INST_CNT = 0x30 (read-only)
   - Define state constants: STATE_IDLE=0, STATE_ACTIVE=1, STATE_DONE=2
   - Define instruction type constants: INST_FENCE=0, INST_MATRIX=1, INST_DMA=2
   - Define MxFunc constants: LOAD_STATIONARY=0, ATTENTION_SCORE_COMPUTE=1,
     ATTENTION_VALUE_COMPUTE=2, ATTENTION_LSE_NORM_SCALE=3, ATTENTION_LSE_NORM=4
   - Define DMAFunc constants: LD_SRAM=0, ST_SRAM=1
   - Define ISA bit-width constants from ISA.Constants:
     I_TYPE_BITS=3, SEM_ID_BITS=5, SEM_VALUE_BITS=3, MX_FUNC_BITS=5,
     DMA_FUNC_BITS=4, DMA_SIZE_BITS=10, DMA_REPEAT_BITS=9,
     SPAD_MAX_ADDR_BITS=20, SPAD_STRIDE_BITS=5, ACC_MAX_ADDR_BITS=20,
     ACC_STRIDE_BITS=5, MEM_MAX_ADDR_BITS=39, MEM_STRIDE_1_BITS=6,
     MEM_STRIDE_2_BITS=15
   - Define MMIO address space size: FSA_MMIO_SIZE = 0x100 (256 bytes)
   - Compute derived values: spadAddrWidth, accAddrWidth, sramAddrWidth

2. Rewrite `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py`:
   - Replace the existing implementation entirely
   - Inherit from `dev.MMIO` (with fallback for standalone mode as before)
   - Implement the hardware-aligned register map:
     - `store()`: handle writes to REG_INST_QUEUE (push 32-bit word to instruction queue),
       REG_SET_ACTIVE (trigger instruction execution)
     - `load()`: handle reads from REG_STATE, all performance counters,
       REG_ENQ_INST_CNT, REG_DEQ_INST_CNT
     - For unimplemented/unused offsets, return 0 on load, ignore on store
   - Maintain an internal instruction queue (list of 32-bit integers)
   - Maintain state machine: idle → active → done
   - Maintain performance counter registers (all uint32)
   - Register with `@dev.register("fsa_mmio", size=0x100)`
   - The `store`/`load` methods use `addr & 0xff` for relative offset within the MMIO space
     (the base address is handled by Spike's bus routing)
   - Do NOT implement instruction execution yet (Phase 2) — just queue instructions and
     update state/counters
   - Keep the file self-contained and importable without pyspike (fallback mode)
   - Add a `test_register_map()` function that verifies:
     - Writing to REG_INST_QUEUE increments enqInstCnt
     - Writing 1 to REG_SET_ACTIVE transitions idle→active
     - Reading REG_STATE returns correct state
     - All perf counters are readable and return 0 initially

**Reference files**:
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/AXI4FSA.scala` (register map: lines 80-94, state machine: lines 96-113)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/isa/ISA.scala` (constants)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/dev.py` (MMIO base class and register decorator)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py` (current implementation to replace)

**Constraints**:
- No external dependencies beyond Python stdlib and numpy
- Must work in both pyspike mode (with dev.MMIO) and standalone mode (without)
- Follow the existing code style in the pyspike project
- Do NOT add comments unless they are critical for understanding
```

### Review Agent Prompt

```
You are reviewing Phase 1 of the PySpike FSA MMIO alignment project.

**Review the following files**:
1. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_config.py`
2. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py`

**Check the following criteria**:

1. **Register map correctness**: Compare every register offset and access type (R/W) against
   AXI4FSA.scala lines 80-94. Every offset must match exactly.

2. **State machine correctness**: The states must be idle(0), active(1), done(2) matching
   AXI4FSA.scala lines 42, 96-113. State transitions:
   - idle → active: when set_active is written
   - active → done: when fence with stop=1 is processed (placeholder for now)
   - done → active: when set_active is written again (reset perf counters)

3. **ISA constants completeness**: All constants from ISA.scala must be present in fsa_config.py.
   Verify bit widths match exactly.

4. **FSAParams correctness**: Verify spadRows = 2*cols + 4*rows, accRows = 1 + rows for
   default configs, matching Configs.scala lines 112-129.

5. **MMIO size**: Must be 0x100 (256 bytes), not 0x10000.

6. **Instruction queue**: Writing to offset 0x00 must append a 32-bit word to the queue.
   enqInstCnt must increment on each write.

7. **PySpike integration**: The @dev.register decorator must use size=0x100.
   The class must properly inherit from dev.MMIO when available.

8. **Standalone mode**: Must work without pyspike installed (fallback base class).

9. **Code quality**: No unnecessary comments, clean structure, follows existing project style.

10. **Test function**: test_register_map() must cover all register offsets and state transitions.

Return a detailed review with:
- PASS/FAIL for each criterion
- Specific issues found with file:line references
- Suggested fixes for any failures
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 1 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria in:
1. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_config.py`
2. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py`

**Guidelines**:
- Make minimal changes to address only the specific issues identified
- Preserve all passing aspects of the implementation
- Re-run the test_register_map() function after fixes to verify
- Do not add comments unless critical
- Maintain compatibility with both pyspike and standalone modes
```

---

## Phase 2: Instruction Decoder

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 2 of the PySpike FSA MMIO alignment project.

**Goal**: Implement a Python instruction decoder that parses the 32-bit instruction words
from the instruction queue into structured instruction objects, matching the hardware Decoder
in Decoder.scala.

**Tasks**:

1. Create `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_decoder.py`:

   Implement instruction decoding that mirrors the Chisel Decoder.scala behavior:

   a) **InstructionMerger**: Collect 32-bit words and merge into complete instructions:
      - Fence: 1 word (instType bits [2:0] == 0)
      - Matrix: 3 words (instType bits [2:0] == 1)
      - DMA: 4 words (instType bits [2:0] == 2)
      - The first word's top 3 bits (bits [31:29] in little-endian 32-bit word, but actually
        bits [2:0] since Chisel UInt head() takes LSBs) determine the instruction type.
        In the 32-bit word, instType occupies the LOWEST 3 bits.

   b) **FenceInstruction** dataclass:
      - instType: int (always 0)
      - matrix: bool (bit 3)
      - dma: bool (bit 4)
      - stop: bool (bit 5)

   c) **MatrixInstruction** dataclass (3 words = 96 bits):
      Parse the concatenated 96-bit value (word0 | word1<<32 | word2<<64) into:
      
      **Header (word 0, 32 bits)**:
      - instType: bits [2:0] (3 bits, always 1)
      - semId: bits [7:3] (5 bits)
      - acquireValid: bit [8] (1 bit)
      - acquireSemValue: bits [11:9] (3 bits)
      - releaseValid: bit [12] (1 bit)
      - releaseSemValue: bits [15:13] (3 bits)
      - func: bits [20:16] (5 bits, MX_FUNC_BITS)
      - waitPrevAcc: bit [21] (1 bit)
      - remaining bits: padding
      
      **Spad (word 1, 32 bits)**:
      - addr: bits [19:0] (up to SPAD_MAX_ADDR_BITS=20)
      - stride: bits [24:20] (5 bits signed, SPAD_STRIDE_BITS)
      - revInput: bit [25]
      - revOutput: bit [26]
      - delayOutput: bit [27]
      - remaining bits: padding
      
      **Acc (word 2, 32 bits)**:
      - addr: bits [19:0] (up to ACC_MAX_ADDR_BITS=20)
      - stride: bits [24:20] (5 bits signed, ACC_STRIDE_BITS)
      - zero: bit [25]
      - remaining bits: padding

   d) **DMAInstruction** dataclass (4 words = 128 bits):
      Parse the concatenated 128-bit value (word0 | word1<<32 | word2<<64 | word3<<96):
      
      **Header (word 0, 32 bits)**:
      - instType: bits [2:0] (3 bits, always 2)
      - semId: bits [7:3] (5 bits)
      - acquireValid: bit [8]
      - acquireSemValue: bits [11:9] (3 bits)
      - releaseValid: bit [12]
      - releaseSemValue: bits [15:13] (3 bits)
      - func: bits [19:16] (4 bits, DMA_FUNC_BITS)
      - repeat: bits [28:20] (9 bits, DMA_REPEAT_BITS)
      - remaining bits: padding
      
      **SRAM (word 1, 32 bits)**:
      - addr: bits [19:0] (up to SRAM_MAX_ADDR_BITS=20)
      - stride: bits [24:20] (5 bits signed, SRAM_STRIDE_BITS)
      - isAccum: bit [25]
      - mem_stride1: bits [31:26] (6 bits, MEM_STRIDE_1_BITS)
      
      **Mem (words 2-3, 64 bits)**:
      - addr: bits [38:0] (up to MEM_MAX_ADDR_BITS=39)
      - stride2: bits [53:39] (15 bits, MEM_STRIDE_2_BITS)
      - size: bits [63:54] (10 bits, DMA_SIZE_BITS)
      
      Computed property: memStride = sign_extend(mem_stride1 ## stride2) (21-bit signed)

   e) **Decoder class**: 
      - Maintains an InstructionMerger state (buffer + count)
      - `feed(word: int) -> Optional[Instruction]`: feed a 32-bit word, return a complete
        instruction when enough words are collected, else None
      - `reset()`: clear the merger state
      - The decoder inspects the instType of the FIRST word to determine how many words
        to collect (matching Decoder.scala lines 46-50)

   f) **Helper functions**:
      - `sign_extend(value: int, bits: int) -> int`: sign-extend a `bits`-width value
      - `bit_slice(value: int, hi: int, lo: int) -> int`: extract bits [hi:lo] inclusive

2. Add a `test_decoder()` function that:
   - Creates known instruction encodings and verifies decoding
   - Tests FenceInstruction: encode stop=1, matrix=1, dma=1 and verify all fields
   - Tests MatrixInstruction with LOAD_STATIONARY func
   - Tests DMAInstruction with LD_SRAM func, specific addr/stride/size values
   - Tests the Decoder.feed() state machine: feed words one at a time, verify None
     returned until complete, then verify the final instruction

**Reference files**:
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/frontend/Decoder.scala` (decoder logic)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/isa/ISA.scala` (constants and bit widths)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/isa/MatrixInstruction.scala` (matrix format)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/isa/DMAInstruction.scala` (DMA format)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/isa/FenceInstruction.scala` (fence format)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_config.py` (constants from Phase 1)

**IMPORTANT NOTE on bit field ordering**:
Chisel Bundles lay out fields in declaration order from LSB to MSB. The `NBytesBundle`
ensures total width = n*8. Padding fields fill remaining bits. When we concatenate
words as word0 | word1<<32 | word2<<64, the Chisel `asTypeOf` cast maps the Bundle
fields onto this concatenated integer with the first declared field at the LSB.
However, within each 32-bit word, the bit layout follows the Bundle declaration order
from LSB to MSB. You must carefully map each field to its correct bit position.

**Constraints**:
- No external dependencies beyond Python stdlib
- Import constants from fsa_config.py
- All dataclasses should be immutable (frozen=True) where possible
- Do not add comments unless critical
```

### Review Agent Prompt

```
You are reviewing Phase 2 of the PySpike FSA MMIO alignment project.

**Review the file**: `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_decoder.py`

**Check the following criteria**:

1. **Bit field extraction correctness**: For each instruction type, verify that bit positions
   match the Chisel Bundle layout. Chisel lays out Bundle fields from LSB to MSB in
   declaration order. Cross-reference with:
   - FenceInstruction.scala: instType(3b), matrix(1b), dma(1b), stop(1b), pad
   - MatrixInstructionHeader: instType(3b), semId(5b), acquireValid(1b), acquireSemValue(3b),
     releaseValid(1b), releaseSemValue(3b), func(5b), waitPrevAcc(1b), pad
   - MatrixInstructionSpad: addr(up to 20b), stride(5b signed), revInput(1b), revOutput(1b),
     delayOutput(1b), pad
   - MatrixInstructionAcc: addr(up to 20b), stride(5b signed), zero(1b), pad
   - DMAInstructionHeader: instType(3b), semId(5b), acquireValid(1b), acquireSemValue(3b),
     releaseValid(1b), releaseSemValue(3b), func(4b), repeat(9b), pad
   - DMAInstructionSRAM: addr(up to 20b), stride(5b signed), isAccum(1b), mem_stride1(6b), pad
   - DMAInstructionMem: addr(up to 39b), stride2(15b), size(10b), pad

2. **Word concatenation order**: The Chisel `asTypeOf` on the InstructionMerger output
   (buf.asUInt) maps the first word to the lowest 32 bits. Verify the decoder concatenates
   words in the same order: word0 is the header (lowest bits).

3. **InstructionMerger behavior**: Must match Decoder.scala:
   - Collects 1/3/4 words based on instType of the FIRST word
   - instType is determined from the lowest 3 bits of the first word
   - Returns None until all words collected

4. **Sign extension**: stride fields must be properly sign-extended from their bit widths.
   memStride must combine mem_stride1 (high bits) and stride2 (low bits) as a 21-bit signed value.

5. **DMA size field**: Must be in bytes, matching DMA_SIZE_BITS=10 (up to 1023 bytes per repeat).

6. **Test coverage**: test_decoder() must cover:
   - All three instruction types
   - Sign-extended stride values (positive and negative)
   - Multi-word assembly via Decoder.feed()
   - Edge cases: repeat=0, addr=0

7. **Code quality**: Clean dataclass definitions, no unnecessary comments, proper use of
   fsa_config constants.

Return a detailed review with PASS/FAIL for each criterion, specific issues with file:line
references, and suggested fixes.
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 2 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria in:
`/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_decoder.py`

**Guidelines**:
- Make minimal changes to address only the specific issues
- Preserve all passing aspects
- Re-run test_decoder() after fixes
- Do not add comments unless critical
- Ensure bit field positions are verified against the Chisel source
```

---

## Phase 3: Scratchpad, Accumulator, and Semaphore Model

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 3 of the PySpike FSA MMIO alignment project.

**Goal**: Implement Python models for the FSA scratchpad memory, accumulator memory, and
semaphore synchronization mechanism.

**Tasks**:

1. Create `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_memory.py`:

   a) **Scratchpad class**: Models the BankedSRAM used for Q/K/V storage
      - Parameterized by: rows (spadRows), rowSize (saRows), elemWidth (16 for FP16)
      - Internal storage: 2D numpy array of shape (rows, rowSize) with dtype matching elemType
      - For FP16: use numpy float16; for FP32 accumulator: use numpy float32
      - `write_row(addr: int, data: np.ndarray)`: write a full row at SRAM address
      - `read_row(addr: int) -> np.ndarray`: read a full row from SRAM address
      - `write_narrow(addr: int, sub_bank_idx: int, data: np.ndarray)`: write one AXI beat
        (narrow write, used by DMA load path)
      - Row size in bytes = rowSize * elemWidth / 8
      - Number of sub-banks per row = rowSize * elemWidth / 8 / beatBytes (beatBytes=4 for 32-bit AXI)
      - Validate address bounds on every access
      - Special constant rows (matching FSA.scala lines 135-160):
        - addr = SpadConstIdx.ONE (0): returns row of 1.0 values
        - addr = SpadConstIdx.AttentionScale (1): returns row of log2(e)/sqrt(dk) values
        - addr = SpadConstIdx.Exp2Slopes (2): returns piecewise-linear exp2 slope values
          (cycle through the slopes array)

   b) **Accumulator class**: Models the BankedSRAM used for accumulation
      - Parameterized by: rows (accRows), rowSize (saCols), elemWidth (32 for FP32)
      - Same interface as Scratchpad but with additional:
      - `read_modify_write(addr: int, data: np.ndarray)`: read current row, apply operation,
        write back (matching the RMW mechanism in FSA.scala lines 184-187)
      - Special constant rows:
        - addr = AccConstIdx.ZERO (0): returns row of 0.0 values
      - Accumulator compute unit support (matching Accumulator.scala):
        - `exp_s1(sa_in, sram_in)`: scale = sa_in * attentionScale + 0
        - `exp_s2(scale)`: scale = 2^scale (piecewise-linear)
        - `acc_sa(scale, sram_in, sa_in)`: out = scale * sram_in + sa_in
        - `acc(scale, sram_in)`: out = scale * sram_in + 0
        - `set_scale(sram_in)`: scale = sram_in
        - `reciprocal(scale)`: compute 1/scale

   c) **Semaphores class**: Models the hardware semaphore mechanism
      - 32 semaphores (SEM_ID_BITS=5), each with 3-bit value (SEM_VALUE_BITS=3)
      - `acquire(sem_id: int, acquire_value: int) -> bool`: returns True if semaphore
        value matches acquire_value AND not busy; sets busy=True on success
      - `release(sem_id: int, release_value: int)`: sets semaphore value, clears busy
      - `is_busy(sem_id: int) -> bool`: check if semaphore is currently acquired
      - `get_value(sem_id: int) -> int`: read current semaphore value
      - `reset()`: clear all semaphores and busy flags

   d) **FSAConfig helper**: A class that holds a complete FSA configuration and creates
      properly-sized Scratchpad and Accumulator instances:
      - Takes FSAParams from fsa_config.py
      - Computes spad/acc dimensions
      - Provides factory methods for creating memory instances

2. Add a `test_memory()` function that:
   - Creates a 4x4 FSA config scratchpad (spadRows=24, rowSize=4, elemWidth=16)
   - Tests write_row/read_row round-trip
   - Tests narrow write (simulating DMA beat writes)
   - Tests constant reads (ONE, AttentionScale, Exp2Slopes)
   - Creates a 4x4 accumulator (accRows=5, rowSize=4, elemWidth=32)
   - Tests accumulator operations: exp_s1, exp_s2, acc_sa, acc, set_scale, reciprocal
   - Tests semaphores: acquire/release, busy state, value persistence
   - Tests semaphore blocking: acquire fails when value doesn't match

**Reference files**:
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/BankedSRAM.scala` (SRAM structure)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/FSA.scala` (SPAD/ACC instantiation: lines 92-112, constants: lines 135-160)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/Accumulator.scala` (accumulator commands)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/frontend/Semaphores.scala` (semaphore logic)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/Configs.scala` (default params)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/arithmetic/FPArithmeticImpl.scala` (FP arithmetic, exp2 constants)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_config.py` (Phase 1 output)

**Constraints**:
- Use numpy for array operations (already a project dependency)
- Import constants from fsa_config.py
- For FP16, use numpy.float16; for FP32, use numpy.float32
- The exp2 piecewise-linear implementation should use the same slopes/intercepts as
  FPArithmeticImpl.scala (you can approximate with numpy.exp2 for the functional model,
  but the slopes/intercepts constants must be defined for constant-row reads)
- Do not add comments unless critical
```

### Review Agent Prompt

```
You are reviewing Phase 3 of the PySpike FSA MMIO alignment project.

**Review the file**: `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_memory.py`

**Check the following criteria**:

1. **Scratchpad dimensions**: For fsa4x4 config, spadRows must be 24 (2*4+4*4),
   rowSize=4 (saRows), elemWidth=16 (FP16). Verify against Configs.scala:131.

2. **Accumulator dimensions**: For fsa4x4, accRows must be 5 (1+4), rowSize=4 (saCols),
   elemWidth=32 (FP32). Verify against Configs.scala:121-128.

3. **Narrow write correctness**: DMA writes one beat (4 bytes = 2 FP16 elements) at a time
   to a specific sub-bank. Verify the sub-bank indexing matches BankedSRAM.scala.

4. **Constant row addresses**: Must match FSA.scala lines 135-160:
   - SpadConstIdx.ONE = 0
   - SpadConstIdx.AttentionScale = 1
   - SpadConstIdx.Exp2Slopes = 2
   - AccConstIdx.ZERO = 0

5. **Accumulator operations**: Each operation must match Accumulator.scala lines 29-81:
   - EXP_S1: scale = sa_in * attentionScale
   - EXP_S2: scale = 2^scale
   - ACC_SA: out = scale * sram_in + sa_in
   - ACC: out = scale * sram_in
   - SET_SCALE: scale = sram_in
   - RECIPROCAL: 1/scale

6. **Semaphore semantics**: Must match Semaphores.scala:
   - Acquire blocks (returns False) when value doesn't match OR busy=True
   - Release sets value and clears busy
   - 32 semaphores, 3-bit values

7. **Read-modify-write**: Accumulator must support RMW for in-place accumulation
   (FSA.scala lines 184-187).

8. **Test coverage**: test_memory() must cover all operations and edge cases.

9. **Code quality**: Clean class design, proper use of numpy, no unnecessary comments.

Return a detailed review with PASS/FAIL for each criterion, specific issues, and fixes.
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 3 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria in:
`/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_memory.py`

**Guidelines**:
- Make minimal changes to address only the specific issues
- Preserve all passing aspects
- Re-run test_memory() after fixes
- Do not add comments unless critical
```

---

## Phase 4: Functional Simulation Engine (FlashAttention)

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 4 of the PySpike FSA MMIO alignment project.

**Goal**: Implement the functional simulation engine that executes FSA instructions using NumPy,
providing a software model of the FlashAttention computation pipeline.

**Tasks**:

1. Create `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_engine.py`:

   a) **FSAEngine class**: The core simulation engine
      - Constructor takes: FSAParams (from fsa_config.py), a memory read/write interface
        (callbacks for DMA to access main memory), and the Scratchpad/Accumulator/Semaphores
        instances from fsa_memory.py
      - Maintains: instruction queue, state (idle/active/done), performance counters
      - Main method: `execute()` — processes all queued instructions until a fence with stop=1

   b) **DMA execution**: When a DMA instruction is decoded:
      - LD_SRAM (func=0): Read `size` bytes from main memory at `memAddr`, write to
        scratchpad/accumulator at `sramAddr`. Repeat `repeat` times with strides.
        For each repeat iteration:
          - Read `size` bytes from memAddr using the memory callback
          - Convert bytes to numpy array (FP16 for scratchpad, FP32 for accumulator)
          - Write to the appropriate SRAM row(s)
          - Advance memAddr by memStride, sramAddr by sramStride
      - ST_SRAM (func=1): Read from accumulator at `sramAddr`, write `size` bytes to
        main memory at `memAddr`. Similar strided repetition.
      - Handle semaphore acquire/release at the appropriate points

   c) **Matrix instruction execution**: When a matrix instruction is decoded:
      - LOAD_STATIONARY (func=0): Load Q from scratchpad into internal SA register
        - Read `cols` rows from scratchpad starting at spad.addr
        - Store as self.sa_registers (2D array: rows x cols, representing Q)
      - ATTENTION_SCORE_COMPUTE (func=1): Compute S = Q @ K^T with online softmax
        - Read K from scratchpad: `rows` rows starting at spad.addr
        - Compute S = Q @ K^T using numpy matmul
        - Online softmax: track row-wise max, compute exp(S - max), sum of exp
        - Store attention weights (P = softmax(S)) internally for next step
        - Update accumulator: store exp-sum in acc row 0, handle old-max/new-max correction
      - ATTENTION_VALUE_COMPUTE (func=2): Compute O = P @ V
        - Read V from scratchpad: `rows` rows starting at spad.addr
        - Compute P @ V using numpy matmul
        - Read old O from accumulator (rows starting at acc.addr)
        - Accumulate: O_new = scale * O_old + P @ V (online softmax correction)
        - Write O_new back to accumulator
      - ATTENTION_LSE_NORM_SCALE (func=3): Compute 1/lse
        - Read lse from accumulator row 0
        - Compute reciprocal: scale = 1.0 / lse
        - Store scale for next instruction
      - ATTENTION_LSE_NORM (func=4): Final normalization O = O * (1/lse)
        - Read O from accumulator
        - Multiply by stored scale: O = scale * O
        - Write normalized O back to accumulator
      - Handle semaphore acquire/release

   d) **Fence instruction execution**:
      - Wait for DMA to complete (if dma flag set)
      - Wait for matrix engine to complete (if matrix flag set)
      - If stop flag set: transition state to done
      - Handle semaphore release

   e) **Memory interface**: The engine needs to access Spike's main memory for DMA operations.
      Define an abstract interface:
      ```python
      class MemoryInterface:
          def read(self, addr: int, size: int) -> bytes: ...
          def write(self, addr: int, data: bytes) -> None: ...
      ```
      Provide a concrete implementation that wraps pyspike's sim_t for memory access,
      and a simple dict-based implementation for standalone testing.

   f) **Performance counters**: Track during execution:
      - execTime: total cycles in active state
      - mxBubble: cycles SA was idle while active
      - mxActive: cycles SA was computing
      - dmaActive: cycles DMA was active
      - rawInst: total raw instructions dequeued
      - mxInst: matrix instructions executed
      - dmaInst: DMA instructions executed
      - fence: fence instructions executed

2. Add a `test_engine()` function that:
   - Creates a 4x4 FSA engine with a simple memory backend
   - Loads Q, K, V matrices into memory (4x4 FP16)
   - Executes a complete FlashAttention sequence:
     1. DMA LD_SRAM: load Q to scratchpad
     2. Matrix LOAD_STATIONARY: load Q into SA
     3. DMA LD_SRAM: load K to scratchpad
     4. Matrix ATTENTION_SCORE_COMPUTE: compute S = Q @ K^T with softmax
     5. DMA LD_SRAM: load V to scratchpad
     6. Matrix ATTENTION_VALUE_COMPUTE: compute O = P @ V
     7. Matrix ATTENTION_LSE_NORM_SCALE: compute 1/lse
     8. Matrix ATTENTION_LSE_NORM: normalize O
     9. DMA ST_SRAM: store O back to memory
     10. Fence with stop=1
   - Compares result with numpy reference: softmax(Q @ K^T) @ V
   - Verifies max absolute error < 0.01 (FP16 precision)

**Reference files**:
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/ExecutionPlan.scala` (execution plan details)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/FSA.scala` (data flow)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/Accumulator.scala` (accumulator operations)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/dma/DMA.scala` (DMA engine)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/dma/LSQ.scala` (load/store queues)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_config.py` (Phase 1)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_decoder.py` (Phase 2)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_memory.py` (Phase 3)

**Key implementation notes**:
- The functional model does NOT need to be cycle-accurate. It should produce the same
  numerical results as the hardware but can use direct numpy operations.
- For online softmax in ATTENTION_SCORE_COMPUTE, the functional model should implement
  the standard online softmax algorithm (not the piecewise-linear exp2 approximation).
  This matches the mathematical intent while being simpler to implement.
- The scale factor in ATTENTION_VALUE_COMPUTE handles the online softmax correction:
  when a new row-max is found, previous exp values need to be rescaled by exp(old_max - new_max).
  In the functional model, you can compute this directly.
- DMA strided access: memStride and sramStride are signed values. Positive stride moves
  forward, negative moves backward.

**Constraints**:
- Use numpy for all matrix operations
- Import from fsa_config, fsa_decoder, fsa_memory
- The MemoryInterface must be abstract enough to work with both pyspike's sim_t
  and a simple test backend
- Do not add comments unless critical
```

### Review Agent Prompt

```
You are reviewing Phase 4 of the PySpike FSA MMIO alignment project.

**Review the file**: `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_engine.py`

**Check the following criteria**:

1. **FlashAttention correctness**: The complete pipeline (LOAD_STATIONARY →
   ATTENTION_SCORE_COMPUTE → ATTENTION_VALUE_COMPUTE → LSE_NORM_SCALE → LSE_NORM)
   must produce results matching numpy's reference implementation of
   softmax(Q @ K^T / sqrt(dk)) @ V within FP16 precision (< 0.01 max error).

2. **Online softmax**: ATTENTION_SCORE_COMPUTE must implement online softmax correctly:
   - Track running row-max
   - Compute exp(S - max) for numerical stability
   - Sum of exp values for normalization
   - Handle the correction factor when max updates (rescale previous sum)

3. **DMA strided access**: Must correctly handle:
   - Positive and negative strides (signed values)
   - Repeat count (strided repetition)
   - isAccum flag (selects accumulator vs scratchpad)
   - Size field (transfer size in bytes per repeat)

4. **Accumulator operations**: Must match the hardware behavior:
   - ACC_SA: out = scale * sram_in + sa_in (accumulate with SA output)
   - ACC: out = scale * sram_in (scale only)
   - Online softmax correction via exp(old_max - new_max) scaling

5. **Semaphore handling**: Instructions with acquireValid must block until the semaphore
   is available. Instructions with releaseValid must release after completion.

6. **State machine**: After fence with stop=1, state must transition to done.
   Performance counters must be reset on done→active transition.

7. **MemoryInterface**: Must be abstract (protocol/ABC) with read/write methods.
   A concrete SimMemoryInterface should wrap pyspike's sim_t.
   A DictMemoryInterface should be provided for testing.

8. **Performance counters**: All 8 counters must be tracked and readable.

9. **Test coverage**: test_engine() must verify end-to-end FlashAttention with numerical
   comparison against numpy reference.

10. **Code quality**: Clean separation of concerns, proper use of numpy, no unnecessary comments.

Return a detailed review with PASS/FAIL for each criterion, specific issues, and fixes.
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 4 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria in:
`/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_engine.py`

**Guidelines**:
- Make minimal changes to address only the specific issues
- Preserve all passing aspects
- Re-run test_engine() after fixes
- For numerical correctness issues, verify against numpy reference implementation
- Do not add comments unless critical
```

---

## Phase 5: Integration — Wire Engine into MMIO Device

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 5 of the PySpike FSA MMIO alignment project.

**Goal**: Integrate the FSA engine (Phase 4) into the MMIO device (Phase 1), creating a
complete PySpike MMIO peripheral that a RISC-V program can drive through the hardware-aligned
register interface.

**Tasks**:

1. Update `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py`:

   a) Import and instantiate the FSAEngine from fsa_engine.py
   b) When a word is written to REG_INST_QUEUE (offset 0x00):
      - Feed the 32-bit word to the Decoder
      - If the decoder returns a complete instruction, add it to the engine's instruction queue
      - Increment enqInstCnt
   c) When 1 is written to REG_SET_ACTIVE (offset 0x04):
      - If state is idle or done: transition to active
      - If state is done: reset performance counters first
      - Execute all queued instructions via the engine
      - After execution completes (fence with stop), transition to done
      - Increment deqInstCnt for each instruction dequeued
   d) Update performance counters from the engine after execution
   e) The `load()` method returns current values of state and perf counters
   f) For the MemoryInterface: when running in pyspike mode, use `self.sim` to access
      Spike's memory. When in standalone mode, use a DictMemoryInterface.
   g) The `tick()` method can be used for incremental execution (process one instruction
      per tick) or left as no-op for batch execution

2. Create `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_sim_memory.py`:
   - `SimMemoryInterface`: wraps pyspike's sim_t for memory access
     - `read(addr, size) -> bytes`: use sim's memory read interface
     - `write(addr, data: bytes)`: use sim's memory write interface
   - Note: pyspike's sim_t exposes memory through the mem layout. You may need to
     access the underlying bus/memory objects. Check the pyspike C++ bindings for
     available memory access methods. If direct memory access is not available through
     the Python API, implement a workaround using a DictMemoryInterface and document
     the limitation.

3. Update the `@dev.register` decorator call to use the correct size (0x100).

4. Add comprehensive integration test `test_integration()`:
   - Create an FSAMMIO device in standalone mode
   - Manually construct instruction words for a complete FlashAttention sequence
   - Write instruction words to REG_INST_QUEUE one at a time
   - Write 1 to REG_SET_ACTIVE
   - Poll REG_STATE until done
   - Read performance counters
   - Verify the computation result by checking the memory backend

**Instruction word construction helper**: Create a helper function or module that can
encode FSA instructions into 32-bit words, so tests don't have to manually construct
bit patterns. This should be the inverse of the decoder:

```python
def encode_fence(matrix: bool, dma: bool, stop: bool) -> list[int]:
    ...

def encode_matrix_instruction(func, spad_addr, spad_stride, acc_addr, acc_stride,
                                sem_id=0, acquire_valid=False, ...) -> list[int]:
    ...

def encode_dma_instruction(func, sram_addr, sram_stride, mem_addr, mem_stride,
                             size, repeat, is_accum=False, ...) -> list[int]:
    ...
```

Add these to fsa_decoder.py or a new fsa_encoder.py file.

**Reference files**:
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py` (Phase 1 output)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_engine.py` (Phase 4 output)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/dev.py` (MMIO base class)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_decoder.py` (Phase 2 output)

**Constraints**:
- The MMIO device must work in both pyspike mode and standalone mode
- Instruction encoding helpers must produce bit-exact encodings that the decoder can round-trip
- Do not add comments unless critical
```

### Review Agent Prompt

```
You are reviewing Phase 5 of the PySpike FSA MMIO alignment project.

**Review the following files**:
1. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py` (updated)
2. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_sim_memory.py` (new)
3. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_encoder.py` (new, if created)

**Check the following criteria**:

1. **MMIO register behavior**: Writing to offset 0x00 must feed words to the decoder.
   Writing 1 to offset 0x04 must trigger execution. Reading offset 0x08 must return state.
   All perf counter offsets must be readable.

2. **Instruction flow**: Words written to REG_INST_QUEUE must be fed to the Decoder.
   Complete instructions must be queued in the engine. REG_SET_ACTIVE must trigger
   execution of all queued instructions.

3. **State transitions**: idle→active on set_active, active→done after fence(stop),
   done→active on set_active (with counter reset). Must match AXI4FSA.scala:96-113.

4. **Encoder-decoder round-trip**: Every instruction encoded by the encoder must decode
   to the same instruction via the decoder. Test all three instruction types.

5. **MemoryInterface integration**: SimMemoryInterface must properly wrap pyspike's
   memory access. DictMemoryInterface must work for standalone testing.

6. **Performance counters**: Must be updated from the engine after execution and readable
   via MMIO loads at the correct offsets.

7. **Integration test**: test_integration() must exercise the complete flow:
   encode instructions → write to MMIO → trigger execution → read state → verify result.

8. **PySpike compatibility**: Must properly inherit from dev.MMIO and use @dev.register
   with size=0x100. Must work with `pyspike --extlib=fsa_mmio --device=fsa_mmio,0x8000`.

9. **Code quality**: Clean integration, no duplicate logic, proper error handling.

Return a detailed review with PASS/FAIL for each criterion, specific issues, and fixes.
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 5 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria in the relevant files.

**Guidelines**:
- Make minimal changes to address only the specific issues
- Preserve all passing aspects
- Re-run test_integration() after fixes
- Do not add comments unless critical
```

---

## Phase 6: RISC-V Driver and Assembly Test Program

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 6 of the PySpike FSA MMIO alignment project.

**Goal**: Write a RISC-V assembly test program that drives the FSA accelerator through the
hardware-aligned MMIO interface, and a Python driver that generates FSA instructions.

**Tasks**:

1. Create `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_driver.py`:
   A Python driver that constructs FSA instruction sequences for FlashAttention:

   a) `FlashAttentionDriver` class:
      - Constructor: takes FSAParams and the MMIO base address
      - `load_q(spad_addr: int, mem_addr: int, rows: int, cols: int) -> list[int]`:
        Generate DMA LD_SRAM instructions to load Q from memory to scratchpad
      - `load_k(spad_addr: int, mem_addr: int, rows: int, cols: int) -> list[int]`:
        Generate DMA LD_SRAM instructions to load K
      - `load_v(spad_addr: int, mem_addr: int, rows: int, cols: int) -> list[int]`:
        Generate DMA LD_SRAM instructions to load V
      - `load_stationary(spad_addr: int, sem_id: int = 0) -> list[int]`:
        Generate Matrix LOAD_STATIONARY instruction
      - `attention_score_compute(spad_addr: int, acc_addr: int, sem_id: int = 0) -> list[int]`:
        Generate Matrix ATTENTION_SCORE_COMPUTE instruction
      - `attention_value_compute(spad_addr: int, acc_addr: int, sem_id: int = 0) -> list[int]`:
        Generate Matrix ATTENTION_VALUE_COMPUTE instruction
      - `attention_lse_norm_scale(acc_addr: int, sem_id: int = 0) -> list[int]`:
        Generate Matrix ATTENTION_LSE_NORM_SCALE instruction
      - `attention_lse_norm(acc_addr: int, sem_id: int = 0) -> list[int]`:
        Generate Matrix ATTENTION_LSE_NORM instruction
      - `store_o(acc_addr: int, mem_addr: int, rows: int, cols: int) -> list[int]`:
        Generate DMA ST_SRAM instructions to store O from accumulator to memory
      - `fence(stop: bool = False, matrix: bool = True, dma: bool = True) -> list[int]`:
        Generate Fence instruction
      - `flash_attention(q_addr, k_addr, v_addr, o_addr, spad_q, spad_k, spad_v, acc_o) -> list[int]`:
        Generate the complete instruction sequence for one FlashAttention inner loop iteration

   b) Use the encoder from Phase 5 to produce 32-bit instruction words
   c) Handle semaphore coordination between DMA and matrix instructions

2. Rewrite `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/asm/fsa_asm.S`:
   A RISC-V assembly program that drives FSA through the hardware-aligned MMIO interface:

   - MMIO base address: 0x8000 (matching AXI4FSA configNode address)
   - Register map (matching AXI4FSA.scala:80-94):
     - 0x00: instruction queue write
     - 0x04: set_active
     - 0x08: state (read)
   - The program should:
     1. Push instruction words for a complete FlashAttention sequence:
        - DMA LD_SRAM for Q, K, V (with semaphore coordination)
        - LOAD_STATIONARY for Q
        - ATTENTION_SCORE_COMPUTE
        - ATTENTION_VALUE_COMPUTE
        - ATTENTION_LSE_NORM_SCALE
        - ATTENTION_LSE_NORM
        - DMA ST_SRAM for O
        - Fence with stop=1
     2. Write 1 to set_active
     3. Poll state register until done (state == 2)
     4. Write 1 to set_active again to reset for next iteration
     5. Loop or exit
   - Use hardcoded instruction word values for a 4x4 FP16 FlashAttention
   - The instruction words should be pre-computed using the encoder

3. Create `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/asm/fsa_asm_encode.py`:
   A Python script that generates the instruction word constants for the assembly program:
   - Uses the FlashAttentionDriver to generate the complete instruction sequence
   - Outputs .equ directives with hex values for each instruction word
   - This script is run at build time to generate constants for the assembly

4. Add `test_driver()` function to fsa_driver.py:
   - Generate a complete FlashAttention instruction sequence
   - Verify each instruction decodes correctly via the decoder
   - Verify the total word count matches expectations

**Reference files**:
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/ExecutionPlan.scala` (execution sequence)
- `/home/zkyd/chipyard-fsa/generators/fsa/src/main/scala/fsa/isa/ISA.scala` (instruction formats)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_encoder.py` (Phase 5 encoder)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_config.py` (Phase 1 config)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/asm/fsa_asm.S` (existing assembly to replace)

**Constraints**:
- The assembly program must be compatible with RISC-V GCC assembler
- Instruction words in the assembly must be pre-computed constants
- The driver must use the encoder for instruction generation
- Do not add comments unless critical
```

### Review Agent Prompt

```
You are reviewing Phase 6 of the PySpike FSA MMIO alignment project.

**Review the following files**:
1. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_driver.py`
2. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/asm/fsa_asm.S`
3. `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/asm/fsa_asm_encode.py`

**Check the following criteria**:

1. **Driver instruction generation**: Each driver method must produce correct instruction
   encodings that the decoder can round-trip. Verify DMA instructions have correct
   func/addr/stride/size/repeat fields.

2. **Semaphore coordination**: DMA loads must release semaphores that matrix instructions
   acquire, ensuring correct execution order. Verify acquire/release values are consistent.

3. **FlashAttention sequence**: The complete sequence must follow the correct order:
   LD_Q → LOAD_STATIONARY → LD_K → ATTENTION_SCORE_COMPUTE → LD_V →
   ATTENTION_VALUE_COMPUTE → ATTENTION_LSE_NORM_SCALE → ATTENTION_LSE_NORM →
   ST_O → Fence(stop)

4. **Assembly correctness**: The assembly must:
   - Use the correct MMIO base address (0x8000)
   - Write instruction words to offset 0x00
   - Write 1 to offset 0x04 to activate
   - Poll offset 0x08 for state == 2
   - Use proper RISC-V instructions (sw for 32-bit store, lw for 32-bit load)

5. **Instruction word values**: The pre-computed instruction words in the assembly must
   match what the encoder produces for the same parameters.

6. **Scratchpad/accumulator addressing**: Q/K/V must be placed at correct scratchpad
   addresses matching the double-buffering scheme (2*cols for Q, 4*rows for K and V).
   Output O must be at accumulator rows 1..rows (row 0 is LSE).

7. **DMA transfer sizes**: Each DMA transfer size must match the matrix dimensions.
   For 4x4 FP16: each row = 4 * 2 = 8 bytes. A tile of 4 rows = 32 bytes.

8. **Code quality**: Clean driver API, correct assembly syntax, no unnecessary comments.

Return a detailed review with PASS/FAIL for each criterion, specific issues, and fixes.
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 6 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria in the relevant files.

**Guidelines**:
- Make minimal changes to address only the specific issues
- Preserve all passing aspects
- Re-run test_driver() after fixes
- Verify assembly instruction words match encoder output
- Do not add comments unless critical
```

---

## Phase 7: End-to-End Test and Demo

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 7 of the PySpike FSA MMIO alignment project.

**Goal**: Create end-to-end tests and a demo that exercises the complete PySpike + FSA
integration, verifying numerical correctness against numpy reference.

**Tasks**:

1. Create `/home/zkyd/chipyard-fsa/pyspike/tests/test_fsa_e2e.py`:
   Comprehensive end-to-end test:

   a) `test_fsa_4x4_fp16()`: Test 4x4 FP16 FlashAttention
      - Create Q, K, V matrices (4x4 FP16) in the memory backend
      - Use FlashAttentionDriver to generate instruction sequence
      - Create FSAMMIO device in standalone mode
      - Write all instruction words to REG_INST_QUEUE
      - Write 1 to REG_SET_ACTIVE
      - Poll REG_STATE until done
      - Read result from memory backend
      - Compare with numpy reference: softmax(Q @ K^T / sqrt(4)) @ V
      - Assert max error < 0.01

   b) `test_fsa_8x8_fp16()`: Same test with 8x8 matrices

   c) `test_fsa_multiple_iterations()`: Test multiple FlashAttention iterations
      (simulating tiling over a longer sequence)
      - Run 2 iterations with different K/V tiles
      - Verify online softmax accumulation across iterations

   d) `test_perf_counters()`: Verify performance counters are non-zero after execution

   e) `test_state_transitions()`: Verify idle→active→done→active→done cycle

   f) `test_fence_stop()`: Verify fence with stop=1 transitions to done state

2. Update `/home/zkyd/chipyard-fsa/pyspike/examples/fsa_pyspike_demo.py`:
   Rewrite the demo to use the new hardware-aligned interface:

   ```python
   #!/usr/bin/env python3
   """FSA PySpike Integration Demo (Hardware-Aligned MMIO)"""

   # Demo shows:
   # 1. Creating FSA MMIO device with hardware-aligned register map
   # 2. Using FlashAttentionDriver to generate instruction sequence
   # 3. Writing instructions via MMIO register interface
   # 4. Triggering execution and polling state
   # 5. Reading results and comparing with numpy reference
   ```

   The demo should:
   - Work in standalone mode (no pyspike required)
   - Use the FlashAttentionDriver to generate instructions
   - Show the complete MMIO interaction flow
   - Print comparison with numpy reference
   - Support both 4x4 and 16x16 configurations

3. Create `/home/zkyd/chipyard-fsa/pyspike/examples/fsa_standalone_demo.py`:
   A simpler standalone demo that doesn't require any pyspike components:
   - Directly uses FSAEngine with DictMemoryInterface
   - Shows the instruction-level programming model
   - Prints each instruction as it's executed
   - Compares result with numpy reference

**Reference files**:
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_mmio.py` (Phase 5 output)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_driver.py` (Phase 6 output)
- `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_engine.py` (Phase 4 output)
- `/home/zkyd/chipyard-fsa/pyspike/examples/fsa_pyspike_demo.py` (existing demo to replace)

**Constraints**:
- All tests must pass without pyspike installed (standalone mode)
- Numerical comparison must use numpy reference implementation
- Demo scripts must be runnable with `python3 <script>`
- Do not add comments unless critical
```

### Review Agent Prompt

```
You are reviewing Phase 7 of the PySpike FSA MMIO alignment project.

**Review the following files**:
1. `/home/zkyd/chipyard-fsa/pyspike/tests/test_fsa_e2e.py`
2. `/home/zkyd/chipyard-fsa/pyspike/examples/fsa_pyspike_demo.py`
3. `/home/zkyd/chipyard-fsa/pyspike/examples/fsa_standalone_demo.py`

**Check the following criteria**:

1. **Numerical correctness**: All e2e tests must verify FlashAttention results against
   numpy reference within FP16 precision (< 0.01 max error for 4x4, < 0.05 for 8x8).

2. **Reference implementation**: The numpy reference must implement:
   scaled_dot_product_attention = softmax(Q @ K^T / sqrt(dk)) @ V
   This is the standard FlashAttention mathematical formula.

3. **Test coverage**: Tests must cover:
   - Different matrix sizes (4x4, 8x8)
   - Multiple iterations (online softmax across tiles)
   - Performance counters
   - State machine transitions
   - Fence with stop

4. **Demo functionality**: Both demos must:
   - Run without errors in standalone mode
   - Show clear output of each step
   - Print numerical comparison with reference
   - Be runnable with python3 directly

5. **Standalone mode**: All tests and demos must work without pyspike installed,
   using DictMemoryInterface and standalone MMIO mode.

6. **Code quality**: Clean test structure, proper assertions, no unnecessary comments.

Return a detailed review with PASS/FAIL for each criterion, specific issues, and fixes.
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 7 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria in the relevant files.

**Guidelines**:
- Make minimal changes to address only the specific issues
- Preserve all passing aspects
- Re-run all e2e tests after fixes
- For numerical issues, verify the numpy reference implementation is correct
- Do not add comments unless critical
```

---

## Phase 8: Cleanup and Documentation

**Status**: ⬜ Not Started

### Build Agent Prompt

```
You are implementing Phase 8 of the PySpike FSA MMIO alignment project.

**Goal**: Clean up the codebase, remove obsolete files, update documentation, and ensure
everything is consistent.

**Tasks**:

1. Remove or update obsolete files:
   - `/home/zkyd/chipyard-fsa/pyspike/src/main/python/riscv/fsa_shmem.py`: This file
     implemented shared-memory IPC with Verilator, which is no longer the primary approach.
     Either remove it or add a deprecation notice pointing to the new fsa_engine.py approach.
   - `/home/zkyd/chipyard-fsa/pyspike/examples/fsa_example.py`: If this references the old
     MMIO interface, update it to use the new interface.

2. Verify all Python modules have proper imports and no circular dependencies:
   - fsa_config.py: no internal imports
   - fsa_decoder.py: imports from fsa_config
   - fsa_memory.py: imports from fsa_config
   - fsa_engine.py: imports from fsa_config, fsa_decoder, fsa_memory
   - fsa_sim_memory.py: imports from fsa_config (optional pyspike dependency)
   - fsa_encoder.py: imports from fsa_config
   - fsa_driver.py: imports from fsa_config, fsa_encoder
   - fsa_mmio.py: imports from fsa_config, fsa_decoder, fsa_engine, fsa_memory

3. Ensure all test functions are discoverable and runnable:
   - Each module's `if __name__ == '__main__'` block should run its test function
   - All tests should pass when run individually

4. Verify the pyspike CLI integration works:
   - `pyspike --extlib=fsa_mmio --device=fsa_mmio,0x8000 program.elf` should register
     the FSA MMIO device at address 0x8000

5. Create a brief usage guide as a docstring in fsa_mmio.py (not a separate file):
   - How to use in standalone mode
   - How to use with pyspike
   - Register map reference
   - Instruction format reference
   - Example code snippets

**Constraints**:
- Do NOT create any README or markdown documentation files
- Do NOT add comments to code unless critical
- Keep the docstring usage guide concise
- All existing tests must still pass after cleanup
```

### Review Agent Prompt

```
You are reviewing Phase 8 of the PySpike FSA MMIO alignment project.

**Review the entire FSA MMIO implementation** for consistency and completeness.

**Check the following criteria**:

1. **No obsolete code**: Old fsa_shmem.py should be removed or clearly deprecated.
   Old register-map references should be gone.

2. **No circular imports**: Verify the import graph is a clean DAG.

3. **All tests pass**: Run every test function and verify they all pass.

4. **Consistent API**: All modules use the same constants from fsa_config.py.
   No hardcoded magic numbers.

5. **PySpike integration**: The device registers correctly with @dev.register("fsa_mmio", size=0x100).
   The CLI invocation `pyspike --extlib=fsa_mmio --device=fsa_mmio,0x8000` should work.

6. **Usage guide**: The docstring in fsa_mmio.py covers standalone and pyspike usage,
   register map, and instruction formats.

7. **Code quality**: No dead code, no unnecessary comments, consistent style across all files.

Return a detailed review with PASS/FAIL for each criterion, specific issues, and fixes.
```

### Fix Agent Prompt

```
You are fixing issues found in Phase 8 of the PySpike FSA MMIO alignment project.

Based on the review feedback, fix all FAIL criteria.

**Guidelines**:
- Make minimal changes to address only the specific issues
- Preserve all passing aspects
- Run all tests after fixes
- Do not add comments unless critical
```

---

## Summary

| Phase | Description | Dependencies | Estimated Complexity |
|-------|-------------|-------------|---------------------|
| 1 | Config & Register Map | None | Low |
| 2 | Instruction Decoder | Phase 1 | Medium |
| 3 | Memory & Semaphore Model | Phase 1 | Medium |
| 4 | Functional Simulation Engine | Phases 1-3 | High |
| 5 | MMIO Integration | Phases 1-4 | Medium |
| 6 | Driver & Assembly Test | Phases 1-5 | Medium |
| 7 | E2E Test & Demo | Phases 1-6 | Medium |
| 8 | Cleanup & Documentation | Phases 1-7 | Low |

Each phase follows the **Build → Review → Fix → Commit** loop. The review agent checks
correctness against the hardware specification, and the fix agent addresses any issues
before committing.
