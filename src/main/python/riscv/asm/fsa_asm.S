.equ FSA_MMIO_BASE,   0x8000
.equ REG_INST_QUEUE,  0x00
.equ REG_SET_ACTIVE,  0x04
.equ REG_STATE,       0x08

.equ STATE_IDLE,      0
.equ STATE_ACTIVE,    1
.equ STATE_DONE,      2

.equ FSA_INST_COUNT, 32
.equ FSA_INST_0,  0x00401002
.equ FSA_INST_1,  0x00100000
.equ FSA_INST_2,  0x80000000
.equ FSA_INST_3,  0x02000400
.equ FSA_INST_4,  0x00003101
.equ FSA_INST_5,  0x00000000
.equ FSA_INST_6,  0x00000000
.equ FSA_INST_7,  0x0040300a
.equ FSA_INST_8,  0x00100004
.equ FSA_INST_9,  0x80000080
.equ FSA_INST_10, 0x02000400
.equ FSA_INST_11, 0x00015309
.equ FSA_INST_12, 0x00000004
.equ FSA_INST_13, 0x00000000
.equ FSA_INST_14, 0x00405012
.equ FSA_INST_15, 0x00100008
.equ FSA_INST_16, 0x80000100
.equ FSA_INST_17, 0x02000400
.equ FSA_INST_18, 0x00027511
.equ FSA_INST_19, 0x00000008
.equ FSA_INST_20, 0x02000001
.equ FSA_INST_21, 0x00037019
.equ FSA_INST_22, 0x00000000
.equ FSA_INST_23, 0x00000000
.equ FSA_INST_24, 0x00040719
.equ FSA_INST_25, 0x00000000
.equ FSA_INST_26, 0x00000000
.equ FSA_INST_27, 0x00410002
.equ FSA_INST_28, 0x02100001
.equ FSA_INST_29, 0x80000200
.equ FSA_INST_30, 0x04000800
.equ FSA_INST_31, 0x00000038

.section .text
.align 2
.globl _start

_start:
    lui   t0, %hi(FSA_MMIO_BASE)
    ori   t0, t0, %lo(FSA_MMIO_BASE)

    la    t1, fsa_inst_table
    li    t2, FSA_INST_COUNT
    li    t3, 0

push_loop:
    slli  t4, t3, 2
    add   t4, t4, t1
    lw    t5, 0(t4)
    sw    t5, REG_INST_QUEUE(t0)
    addi  t3, t3, 1
    bne   t3, t2, push_loop

    li    t4, 1
    sw    t4, REG_SET_ACTIVE(t0)

poll_loop:
    lw    t5, REG_STATE(t0)
    li    t6, STATE_DONE
    beq   t5, t6, done
    j     poll_loop

done:
    li    t4, 1
    sw    t4, REG_SET_ACTIVE(t0)

    li    t3, 0
    la    t1, fsa_inst_table
    j     push_loop

.section .data
.align 2
fsa_inst_table:
    .word FSA_INST_0
    .word FSA_INST_1
    .word FSA_INST_2
    .word FSA_INST_3
    .word FSA_INST_4
    .word FSA_INST_5
    .word FSA_INST_6
    .word FSA_INST_7
    .word FSA_INST_8
    .word FSA_INST_9
    .word FSA_INST_10
    .word FSA_INST_11
    .word FSA_INST_12
    .word FSA_INST_13
    .word FSA_INST_14
    .word FSA_INST_15
    .word FSA_INST_16
    .word FSA_INST_17
    .word FSA_INST_18
    .word FSA_INST_19
    .word FSA_INST_20
    .word FSA_INST_21
    .word FSA_INST_22
    .word FSA_INST_23
    .word FSA_INST_24
    .word FSA_INST_25
    .word FSA_INST_26
    .word FSA_INST_27
    .word FSA_INST_28
    .word FSA_INST_29
    .word FSA_INST_30
    .word FSA_INST_31
