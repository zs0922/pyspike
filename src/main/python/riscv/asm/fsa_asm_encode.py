from __future__ import annotations

from riscv.fsa_config import FSAParams, fsa4x4
from riscv.fsa_driver import FlashAttentionDriver


def generate_asm_equ(params: FSAParams) -> str:
    driver = FlashAttentionDriver(params, mmio_base=0x8000)
    words = driver.flash_attention(
        q_addr=0x80000000,
        k_addr=0x80000080,
        v_addr=0x80000100,
        o_addr=0x80000200,
        spad_q=0,
        spad_k=4,
        spad_v=8,
        acc_o=1,
        acc_lse=0,
    )
    lines = []
    lines.append(f".equ FSA_INST_COUNT, {len(words)}")
    for i, w in enumerate(words):
        lines.append(f".equ FSA_INST_{i}, 0x{w & 0xFFFFFFFF:08x}")
    return "\n".join(lines)


if __name__ == "__main__":
    params = fsa4x4()
    output = generate_asm_equ(params)
    print(output)
