section .code
    align 16

section .text

global _read_sse

;void __stdcall read_sse(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [esp+04] PIXEL_YC       *src
;  [esp+08] uint32_t        size
;  [esp+12] uint32_t        count_n
;)

    _read_sse:
        push ebp
        push edi
        push esi
        push ebx
; @+16

        mov edi, 128
        mov esi, [esp+16+12]; count_n
        mov eax, [esp+16+04]; src
        mov ebp, [esp+16+08]; size
        shr ebp, 7
        align 16
    .OUTER_LOOP:
        mov ebx, eax; src
        mov edx, ebx
        add edx, 64
        mov ecx, ebp
    .INNER_LOOP:
        movaps xmm0, [ebx];
        movaps xmm1, [ebx+16];
        movaps xmm2, [ebx+32];
        movaps xmm3, [ebx+48];
        add ebx, edi;
        movaps xmm4, [edx];
        movaps xmm5, [edx+16];
        movaps xmm6, [edx+32];
        movaps xmm7, [edx+48];
        add edx, edi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP

        pop ebx
        pop esi
        pop edi
        pop ebp

        ret



        
global _read_avx

;void __stdcall read_avx(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [esp+04] PIXEL_YC       *src
;  [esp+08] uint32_t        size
;  [esp+12] uint32_t        count_n
;)

    _read_avx:
        push ebp
        push edi
        push esi
        push ebx
; @+16

        mov edi, 256
        mov esi, [esp+16+12]; count_n
        mov eax, [esp+16+04]; src
        mov ebp, [esp+16+08]; size
        shr ebp, 8
        align 16
    .OUTER_LOOP:
        mov ebx, eax; src
        mov edx, ebx
        add edx, 128
        mov ecx, ebp
    .INNER_LOOP:
        vmovaps ymm0, [ebx]
        vmovaps ymm1, [ebx+32]
        vmovaps ymm2, [ebx+64]
        vmovaps ymm3, [ebx+96]
        add ebx, edi
        vmovaps ymm4, [edx]
        vmovaps ymm5, [edx+32]
        vmovaps ymm6, [edx+64]
        vmovaps ymm7, [edx+96]
        add edx, edi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP

        vzeroupper

        pop ebx
        pop esi
        pop edi
        pop ebp

        ret


global _write_sse

;void __stdcall write_sse(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [esp+04] PIXEL_YC       *src
;  [esp+08] uint32_t        size
;  [esp+12] uint32_t        count_n
;)

    _write_sse:
        push ebp
        push edi
        push esi
        push ebx
; @+16

        mov edi, 128
        mov esi, [esp+16+12]; count_n
        mov eax, [esp+16+04]; src
        mov ebp, [esp+16+08]; size
        shr ebp, 7
        align 16
    .OUTER_LOOP:
        mov ebx, eax; src
        mov edx, ebx
        add edx, 64
        mov ecx, ebp
    .INNER_LOOP:
        movaps [ebx],    xmm0 
        movaps [ebx+16], xmm0 
        movaps [ebx+32], xmm0
        movaps [ebx+48], xmm0
        add ebx, edi
        movaps [edx],    xmm0 
        movaps [edx+16], xmm0 
        movaps [edx+32], xmm0
        movaps [edx+48], xmm0
        add edx, edi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP

        pop ebx
        pop esi
        pop edi
        pop ebp

        ret



global _write_avx

;void __stdcall write_avx(uint8_t *src, uint32_t size, uint32_t count_n) (
;  [esp+04] PIXEL_YC       *src
;  [esp+08] uint32_t        size
;  [esp+12] uint32_t        count_n
;)

_write_avx:
        push ebp
        push edi
        push esi
        push ebx
; @+16

        mov edi, 256
        mov esi, [esp+16+12]; count_n
        mov eax, [esp+16+04]; src
        mov ebp, [esp+16+08]; size
        shr ebp, 8
        align 16
    .OUTER_LOOP:
        mov ebx, eax; src
        mov edx, ebx
        add edx, 128
        mov ecx, ebp
    .INNER_LOOP:
        vmovaps [ebx],    ymm0 
        vmovaps [ebx+32], ymm0 
        vmovaps [ebx+64], ymm0
        vmovaps [eax+96], ymm0
        add ebx, edi
        vmovaps [edx],    ymm0 
        vmovaps [edx+32], ymm0 
        vmovaps [edx+64], ymm0
        vmovaps [edx+96], ymm0
        add edx, edi
        dec ecx
        jnz .INNER_LOOP

        dec esi
        jnz .OUTER_LOOP

        vzeroupper

        pop ebx
        pop esi
        pop edi
        pop ebp

        ret

