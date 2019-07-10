section .code
    align 16

section .text

global _runl_por

;void __stdcall runl_por(uint32_t count_n) (
;  [esp+04] uint32_t        size
;)

    _runl_por:
        push ebp
        push edi
        push esi
        push ebx
; @+16
        xor eax, eax
		cpuid
		rdtscp
		mov esi, eax
		mov edi, edx
        mov ecx, [esp+16+04]; size
        align 16
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
        dec ecx
        jnz .LOOP

		rdtscp
		sub edx, edi
		sbb eax, esi

        pop ebx
        pop esi
        pop edi
        pop ebp

        ret

