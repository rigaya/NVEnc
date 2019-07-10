section .code
    align 16

section .text

global runl_por

;int64_t __stdcall runl_por(uint32_t count_n) (
;  [esp+04] uint32_t        size
;)

    runl_por:
        push rbx
        push rsi
        mov esi, ecx
        mov r9, rsp
        sub rsp, 16*10
        and rsp, -32
        movaps [rsp+16*0], xmm6
        movaps [rsp+16*1], xmm7
        movaps [rsp+16*2], xmm8
        movaps [rsp+16*3], xmm9
        movaps [rsp+16*4], xmm10
        movaps [rsp+16*5], xmm11
        movaps [rsp+16*6], xmm12
        movaps [rsp+16*7], xmm13
        movaps [rsp+16*8], xmm14
        movaps [rsp+16*9], xmm15

        xor rax, rax
        cpuid
        align 16
        rdtscp
        shl rdx, 32
        or rdx, rax
        mov r8, rdx
    .LOOP:
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        por xmm0, xmm0
        dec esi
        jnz .LOOP

        rdtscp
        shl rdx, 32
        or rax, rdx

        sub rax, r8

        movaps xmm6,  [rsp+16*0]
        movaps xmm7,  [rsp+16*1]
        movaps xmm8,  [rsp+16*2]
        movaps xmm9,  [rsp+16*3]
        movaps xmm10, [rsp+16*4]
        movaps xmm11, [rsp+16*5]
        movaps xmm12, [rsp+16*6]
        movaps xmm13, [rsp+16*7]
        movaps xmm14, [rsp+16*8]
        movaps xmm15, [rsp+16*9]

        mov rsp, r9

        pop rsi
        pop rbx

        ret

