import math
import numpy as np

from riscv.fsa_config import (
    FSAParams,
    SPAD_CONST_ONE,
    SPAD_CONST_ATTENTION_SCALE,
    SPAD_CONST_EXP2_SLOPES,
    ACC_CONST_ZERO,
    ACC_CMD_EXP_S1,
    ACC_CMD_EXP_S2,
    ACC_CMD_ACC_SA,
    ACC_CMD_ACC,
    ACC_CMD_SET_SCALE,
    ACC_CMD_RECIPROCAL,
    N_SEMAPHORES,
    SEM_ID_BITS,
    SEM_VALUE_BITS,
)

EXP2_PWL_PIECES = 8

FP16_EXP2_PWL_SLOPES = np.array([
    0.362060546875,
    0.394775390625,
    0.430419921875,
    0.469482421875,
    0.51220703125,
    0.55810546875,
    0.60888671875,
    0.6640625,
], dtype=np.float16)

FP32_EXP2_PWL_SLOPES = np.array([
    0.362060546875,
    0.394775390625,
    0.430419921875,
    0.469482421875,
    0.51220703125,
    0.55810546875,
    0.60888671875,
    0.6640625,
], dtype=np.float32)


def _attention_scale(dk: int, dtype=np.float16):
    return dtype(math.log2(math.e) / math.sqrt(dk))


class Scratchpad:
    def __init__(self, rows: int, rowSize: int, elemWidth: int, beatBytes: int = 4, dk: int = 64):
        self.rows = rows
        self.rowSize = rowSize
        self.elemWidth = elemWidth
        self.beatBytes = beatBytes
        self.dk = dk
        if elemWidth == 16:
            self.elemType = np.float16
        elif elemWidth == 32:
            self.elemType = np.float32
        else:
            self.elemType = np.float64
        self.storage = np.zeros((rows, rowSize), dtype=self.elemType)
        self.rowBytes = rowSize * elemWidth // 8
        self.nSubBanks = self.rowBytes // beatBytes
        self._exp2_slope_counter = 0

    def write_row(self, addr: int, data: np.ndarray):
        assert 0 <= addr < self.rows, f"Scratchpad addr {addr} out of range [0, {self.rows})"
        self.storage[addr] = np.asarray(data, dtype=self.elemType)[:self.rowSize]

    def read_row(self, addr: int) -> np.ndarray:
        if addr == SPAD_CONST_ONE:
            return np.ones(self.rowSize, dtype=self.elemType)
        if addr == SPAD_CONST_ATTENTION_SCALE:
            return np.full(self.rowSize, _attention_scale(self.dk, self.elemType), dtype=self.elemType)
        if addr == SPAD_CONST_EXP2_SLOPES:
            if self.elemWidth == 16:
                slopes = FP16_EXP2_PWL_SLOPES
            else:
                slopes = FP32_EXP2_PWL_SLOPES
            val = slopes[self._exp2_slope_counter % EXP2_PWL_PIECES]
            self._exp2_slope_counter += 1
            return np.full(self.rowSize, val, dtype=self.elemType)
        assert 0 <= addr < self.rows, f"Scratchpad addr {addr} out of range [0, {self.rows})"
        return self.storage[addr].copy()

    def write_narrow(self, addr: int, sub_bank_idx: int, data: np.ndarray):
        assert 0 <= addr < self.rows, f"Scratchpad addr {addr} out of range [0, {self.rows})"
        assert 0 <= sub_bank_idx < self.nSubBanks, f"Sub-bank {sub_bank_idx} out of range [0, {self.nSubBanks})"
        elems_per_subbank = self.rowSize // self.nSubBanks
        start = sub_bank_idx * elems_per_subbank
        end = start + elems_per_subbank
        self.storage[addr, start:end] = np.asarray(data, dtype=self.elemType)[:elems_per_subbank]

    def reset_exp2_counter(self):
        self._exp2_slope_counter = 0


class Accumulator:
    def __init__(self, rows: int, rowSize: int, elemWidth: int, beatBytes: int = 4, dk: int = 64):
        self.rows = rows
        self.rowSize = rowSize
        self.elemWidth = elemWidth
        self.beatBytes = beatBytes
        self.dk = dk
        if elemWidth == 32:
            self.elemType = np.float32
        elif elemWidth == 16:
            self.elemType = np.float16
        else:
            self.elemType = np.float64
        self.storage = np.zeros((rows, rowSize), dtype=self.elemType)
        self.rowBytes = rowSize * elemWidth // 8
        self.nSubBanks = self.rowBytes // beatBytes
        self.scale = np.zeros(rowSize, dtype=self.elemType)

    def write_row(self, addr: int, data: np.ndarray):
        assert 0 <= addr < self.rows, f"Accumulator addr {addr} out of range [0, {self.rows})"
        self.storage[addr] = np.asarray(data, dtype=self.elemType)[:self.rowSize]

    def read_row(self, addr: int) -> np.ndarray:
        if addr == ACC_CONST_ZERO:
            return np.zeros(self.rowSize, dtype=self.elemType)
        assert 0 <= addr < self.rows, f"Accumulator addr {addr} out of range [0, {self.rows})"
        return self.storage[addr].copy()

    def write_narrow(self, addr: int, sub_bank_idx: int, data: np.ndarray):
        assert 0 <= addr < self.rows, f"Accumulator addr {addr} out of range [0, {self.rows})"
        assert 0 <= sub_bank_idx < self.nSubBanks, f"Sub-bank {sub_bank_idx} out of range [0, {self.nSubBanks})"
        elems_per_subbank = self.rowSize // self.nSubBanks
        start = sub_bank_idx * elems_per_subbank
        end = start + elems_per_subbank
        self.storage[addr, start:end] = np.asarray(data, dtype=self.elemType)[:elems_per_subbank]

    def read_modify_write(self, addr: int, data: np.ndarray):
        current = self.read_row(addr)
        self.write_row(addr, data)
        return current

    def exp_s1(self, sa_in: np.ndarray, sram_in: np.ndarray) -> np.ndarray:
        attention_scale_val = _attention_scale(self.dk, self.elemType)
        self.scale = sa_in.astype(self.elemType) * attention_scale_val
        return self.scale.copy()

    def exp_s2(self, scale: np.ndarray) -> np.ndarray:
        self.scale = np.exp2(scale.astype(self.elemType))
        return self.scale.copy()

    def acc_sa(self, scale: np.ndarray, sram_in: np.ndarray, sa_in: np.ndarray) -> np.ndarray:
        out = scale.astype(self.elemType) * sram_in.astype(self.elemType) + sa_in.astype(self.elemType)
        return out

    def acc(self, scale: np.ndarray, sram_in: np.ndarray) -> np.ndarray:
        out = scale.astype(self.elemType) * sram_in.astype(self.elemType)
        return out

    def set_scale(self, sram_in: np.ndarray):
        self.scale = np.asarray(sram_in, dtype=self.elemType)

    def reciprocal(self, scale: np.ndarray) -> np.ndarray:
        self.scale = (1.0 / scale.astype(np.float64)).astype(self.elemType)
        return self.scale.copy()


class Semaphores:
    def __init__(self):
        self.values = [0] * N_SEMAPHORES
        self.busy = [False] * N_SEMAPHORES

    def acquire(self, sem_id: int, acquire_value: int) -> bool:
        assert 0 <= sem_id < N_SEMAPHORES, f"Semaphore id {sem_id} out of range [0, {N_SEMAPHORES})"
        if self.busy[sem_id]:
            return False
        if self.values[sem_id] != acquire_value:
            return False
        self.busy[sem_id] = True
        return True

    def release(self, sem_id: int, release_value: int):
        assert 0 <= sem_id < N_SEMAPHORES, f"Semaphore id {sem_id} out of range [0, {N_SEMAPHORES})"
        self.busy[sem_id] = False
        self.values[sem_id] = release_value

    def is_busy(self, sem_id: int) -> bool:
        assert 0 <= sem_id < N_SEMAPHORES, f"Semaphore id {sem_id} out of range [0, {N_SEMAPHORES})"
        return self.busy[sem_id]

    def get_value(self, sem_id: int) -> int:
        assert 0 <= sem_id < N_SEMAPHORES, f"Semaphore id {sem_id} out of range [0, {N_SEMAPHORES})"
        return self.values[sem_id]

    def reset(self):
        self.values = [0] * N_SEMAPHORES
        self.busy = [False] * N_SEMAPHORES


class FSAConfig:
    def __init__(self, params: FSAParams, elemWidth: int = 16, accElemWidth: int = 32, beatBytes: int = 4):
        self.params = params
        self.elemWidth = elemWidth
        self.accElemWidth = accElemWidth
        self.beatBytes = beatBytes

    def create_scratchpad(self) -> Scratchpad:
        return Scratchpad(
            rows=self.params.spadRows,
            rowSize=self.params.saRows,
            elemWidth=self.elemWidth,
            beatBytes=self.beatBytes,
            dk=self.params.saRows,
        )

    def create_accumulator(self) -> Accumulator:
        return Accumulator(
            rows=self.params.accRows,
            rowSize=self.params.saCols,
            elemWidth=self.accElemWidth,
            beatBytes=self.beatBytes,
            dk=self.params.saRows,
        )

    def create_semaphores(self) -> Semaphores:
        return Semaphores()


def test_memory():
    errors = []

    spad = Scratchpad(rows=24, rowSize=4, elemWidth=16, dk=4)
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    spad.write_row(5, data)
    readback = spad.read_row(5)
    assert np.array_equal(readback, data), f"write_row/read_row mismatch: {readback} != {data}"

    spad2 = Scratchpad(rows=24, rowSize=4, elemWidth=16, dk=4)
    spad2.write_narrow(5, 0, np.array([1.5, 2.5], dtype=np.float16))
    spad2.write_narrow(5, 1, np.array([3.5, 4.5], dtype=np.float16))
    row = spad2.read_row(5)
    expected = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float16)
    assert np.array_equal(row, expected), f"narrow write mismatch: {row} != {expected}"

    one_row = spad.read_row(SPAD_CONST_ONE)
    assert np.all(one_row == 1.0), f"ONE constant mismatch: {one_row}"

    scale_row = spad.read_row(SPAD_CONST_ATTENTION_SCALE)
    expected_scale = np.float16(math.log2(math.e) / math.sqrt(4))
    assert np.all(scale_row == expected_scale), f"AttentionScale mismatch: {scale_row} != {expected_scale}"

    spad.reset_exp2_counter()
    slope_rows = [spad.read_row(SPAD_CONST_EXP2_SLOPES) for _ in range(EXP2_PWL_PIECES)]
    for i, sr in enumerate(slope_rows):
        assert np.all(sr == FP16_EXP2_PWL_SLOPES[i]), f"Exp2Slopes[{i}] mismatch: {sr[0]} != {FP16_EXP2_PWL_SLOPES[i]}"
    wrap_row = spad.read_row(SPAD_CONST_EXP2_SLOPES)
    assert np.all(wrap_row == FP16_EXP2_PWL_SLOPES[0]), f"Exp2Slopes wrap mismatch"

    acc = Accumulator(rows=5, rowSize=4, elemWidth=32, dk=4)
    acc_data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    acc.write_row(1, acc_data)
    acc_read = acc.read_row(1)
    assert np.array_equal(acc_read, acc_data), f"acc write/read mismatch"

    zero_row = acc.read_row(ACC_CONST_ZERO)
    assert np.all(zero_row == 0.0), f"ZERO constant mismatch"

    sa_in = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sram_in = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    scale = acc.exp_s1(sa_in, sram_in)
    expected_scale_val = np.float32(math.log2(math.e) / 2.0)
    expected_scale_arr = sa_in * expected_scale_val
    assert np.allclose(scale, expected_scale_arr, atol=1e-6), f"exp_s1 mismatch: {scale} != {expected_scale_arr}"

    scale2 = acc.exp_s2(scale)
    expected_exp2 = np.exp2(expected_scale_arr)
    assert np.allclose(scale2, expected_exp2, atol=1e-5), f"exp_s2 mismatch: {scale2} != {expected_exp2}"

    acc_sa_out = acc.acc_sa(scale2, sram_in, sa_in)
    expected_acc_sa = scale2 * sram_in + sa_in
    assert np.allclose(acc_sa_out, expected_acc_sa, atol=1e-4), f"acc_sa mismatch"

    acc_out = acc.acc(scale2, sram_in)
    expected_acc = scale2 * sram_in
    assert np.allclose(acc_out, expected_acc, atol=1e-4), f"acc mismatch"

    acc.set_scale(sram_in)
    assert np.array_equal(acc.scale, sram_in), f"set_scale mismatch"

    acc.write_narrow(2, 0, np.array([100.0], dtype=np.float32))
    acc.write_narrow(2, 1, np.array([200.0], dtype=np.float32))
    acc.write_narrow(2, 2, np.array([300.0], dtype=np.float32))
    acc.write_narrow(2, 3, np.array([400.0], dtype=np.float32))
    narrow_row = acc.read_row(2)
    assert np.array_equal(narrow_row, np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)), f"acc write_narrow mismatch: {narrow_row}"

    old_data = np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float32)
    acc.write_row(3, old_data)
    new_data = np.array([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
    old_val = acc.read_modify_write(3, new_data)
    assert np.array_equal(old_val, old_data), f"read_modify_write should return old value: {old_val} != {old_data}"
    assert np.array_equal(acc.read_row(3), new_data), f"read_modify_write should write new value"

    spad_const = Scratchpad(rows=24, rowSize=4, elemWidth=16, dk=4)
    spad_const.write_row(SPAD_CONST_ONE, np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float16))
    one_after = spad_const.read_row(SPAD_CONST_ONE)
    assert np.all(one_after == 1.0), f"CONST_ONE should not be corrupted by write: {one_after}"
    spad_const.write_row(SPAD_CONST_ATTENTION_SCALE, np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float16))
    scale_after = spad_const.read_row(SPAD_CONST_ATTENTION_SCALE)
    expected_as = np.float16(math.log2(math.e) / math.sqrt(4))
    assert np.all(scale_after == expected_as), f"CONST_ATTENTION_SCALE should not be corrupted: {scale_after}"

    sem_bounds = Semaphores()
    sem_bounds.release(0, 7)
    assert sem_bounds.get_value(0) == 7, "semaphore should accept value 7 (3-bit max)"
    sem_bounds.release(0, 0)
    assert sem_bounds.get_value(0) == 0, "semaphore should accept value 0 (3-bit min)"
    sem_bounds.release(1, 5)
    assert sem_bounds.get_value(1) == 5, "semaphore should accept mid-range value 5"

    from riscv.fsa_config import fsa4x4
    cfg = FSAConfig(fsa4x4(), elemWidth=16, accElemWidth=32, beatBytes=4)
    sp = cfg.create_scratchpad()
    assert isinstance(sp, Scratchpad), "create_scratchpad should return Scratchpad"
    assert sp.rows == fsa4x4().spadRows, "scratchpad rows mismatch"
    ac = cfg.create_accumulator()
    assert isinstance(ac, Accumulator), "create_accumulator should return Accumulator"
    assert ac.rows == fsa4x4().accRows, "accumulator rows mismatch"
    se = cfg.create_semaphores()
    assert isinstance(se, Semaphores), "create_semaphores should return Semaphores"

    recip_input = np.array([2.0, 4.0, 0.5, 1.0], dtype=np.float32)
    recip = acc.reciprocal(recip_input)
    expected_recip = np.array([0.5, 0.25, 2.0, 1.0], dtype=np.float32)
    assert np.allclose(recip, expected_recip, atol=1e-6), f"reciprocal mismatch: {recip} != {expected_recip}"
    assert np.allclose(acc.scale, expected_recip, atol=1e-6), f"reciprocal should update self.scale: {acc.scale} != {expected_recip}"

    sem = Semaphores()
    assert not sem.is_busy(0), "semaphore should not be busy initially"
    assert sem.get_value(0) == 0, "semaphore value should be 0 initially"

    assert sem.acquire(0, 0), "acquire with matching value should succeed"
    assert sem.is_busy(0), "semaphore should be busy after acquire"
    assert not sem.acquire(0, 0), "acquire on busy semaphore should fail"

    sem.release(0, 1)
    assert not sem.is_busy(0), "semaphore should not be busy after release"
    assert sem.get_value(0) == 1, "semaphore value should be 1 after release"

    assert not sem.acquire(0, 0), "acquire with wrong value should fail"
    assert sem.acquire(0, 1), "acquire with matching value should succeed"

    sem.reset()
    assert not sem.is_busy(0), "semaphore should not be busy after reset"
    assert sem.get_value(0) == 0, "semaphore value should be 0 after reset"

    print("All tests passed!")


if __name__ == "__main__":
    test_memory()
