#include "blis.h"
#include "bli_avx512_macros.h"
#include <assert.h>

/*
 * Current panel of A: Column-major
 * Current panel of B: Array of blocks. There are k/4 blocks. Each block is a contiguous column-major, 4 by n_r
 *  (For a less efficient implementation, B could be column-major)
 * Current block of C: Column-major 
 */
#define NR 24

#define LOOP_ALIGN ALIGN16

#define UPDATE_C_FOUR_COLS(C1,C2,C3,C4) \
    VMULPS(ZMM(C1), ZMM(C1), ZMM(0)) \
    VMULPS(ZMM(C2), ZMM(C2), ZMM(0)) \
    VMULPS(ZMM(C3), ZMM(C3), ZMM(0)) \
    VMULPS(ZMM(C4), ZMM(C4), ZMM(0)) \
    VFMADD231PS(ZMM(C1), ZMM(1), MEM(RCX      )) \
    VFMADD231PS(ZMM(C2), ZMM(1), MEM(RCX,R12,1)) \
    VFMADD231PS(ZMM(C3), ZMM(1), MEM(RCX,R12,2)) \
    VFMADD231PS(ZMM(C4), ZMM(1), MEM(RCX,R13,1)) \
    VMOVUPS(MEM(RCX      ), ZMM(C1)) \
    VMOVUPS(MEM(RCX,R12,1), ZMM(C2)) \
    VMOVUPS(MEM(RCX,R12,2), ZMM(C3)) \
    VMOVUPS(MEM(RCX,R13,1), ZMM(C4)) \
    LEA(RCX, MEM(RCX,R12,4))

void sgemm_knm_asm_16x24
     (
       dim_t               k_,
       float*    restrict alpha,
       float*    restrict a,
       float*    restrict b,
       float*    restrict beta,
       float*    restrict c, inc_t rs_c, inc_t cs_c
     )
{
    int64_t k = k_ / 4;

    assert(rs_c == 1);
    assert(k_ % 4 == 0);

    __asm__ volatile
    (
        VPXORD(ZMM(8), ZMM(8), ZMM(8)) //clear out registers
        VMOVAPS(ZMM( 9), ZMM(8))   MOV(R12, VAR(cs_c))
        VMOVAPS(ZMM(10), ZMM(8))   MOV(RSI, VAR(k)) //loop index
        VMOVAPS(ZMM(11), ZMM(8))   MOV(RAX, VAR(a)) //load address of a
        VMOVAPS(ZMM(12), ZMM(8))   MOV(RBX, VAR(b)) //load address of b
        VMOVAPS(ZMM(13), ZMM(8))   MOV(RCX, VAR(c)) //load address of c
        VMOVAPS(ZMM(14), ZMM(8)) 
        VMOVAPS(ZMM(15), ZMM(8))   LEA(R12, MEM(,R12,4))    //R12 = cs_c*sizeof(float)
        VMOVAPS(ZMM(16), ZMM(8)) 
        VMOVAPS(ZMM(17), ZMM(8))   
        VMOVAPS(ZMM(18), ZMM(8))   
        VMOVAPS(ZMM(19), ZMM(8))   
        VMOVAPS(ZMM(20), ZMM(8))
        VMOVAPS(ZMM(21), ZMM(8))   LEA(R13, MEM(R12,R12,2)) //R13 = 3*cs_c*sizeof(float)
        VMOVAPS(ZMM(22), ZMM(8))
        VMOVAPS(ZMM(23), ZMM(8))
        VMOVAPS(ZMM(24), ZMM(8))   
        VMOVAPS(ZMM(25), ZMM(8))
        VMOVAPS(ZMM(26), ZMM(8)) 
        VMOVAPS(ZMM(27), ZMM(8)) 
        VMOVAPS(ZMM(28), ZMM(8)) 
        VMOVAPS(ZMM(29), ZMM(8))
        VMOVAPS(ZMM(30), ZMM(8))
        VMOVAPS(ZMM(31), ZMM(8))

        LOOP_ALIGN
        LABEL(MAIN_LOOP)
            // Load 4 vectors of A
            VMOVAPS(ZMM(0), MEM(RAX,(0)*64))
            VMOVAPS(ZMM(1), MEM(RAX,(1)*64))
            VMOVAPS(ZMM(2), MEM(RAX,(2)*64))
            VMOVAPS(ZMM(3), MEM(RAX,(3)*64))

            // Perform a sequence of 24 16x4 MVMs
            V4FMADDPS(ZMM( 8), ZMM(0), MEM(RBX, ( 0)*16))
            V4FMADDPS(ZMM( 9), ZMM(0), MEM(RBX, ( 1)*16))
            V4FMADDPS(ZMM(10), ZMM(0), MEM(RBX, ( 2)*16))
            V4FMADDPS(ZMM(11), ZMM(0), MEM(RBX, ( 3)*16))
            V4FMADDPS(ZMM(12), ZMM(0), MEM(RBX, ( 4)*16))
            V4FMADDPS(ZMM(13), ZMM(0), MEM(RBX, ( 5)*16))
            V4FMADDPS(ZMM(14), ZMM(0), MEM(RBX, ( 6)*16))
            V4FMADDPS(ZMM(15), ZMM(0), MEM(RBX, ( 7)*16))
            V4FMADDPS(ZMM(16), ZMM(0), MEM(RBX, ( 8)*16))
            V4FMADDPS(ZMM(17), ZMM(0), MEM(RBX, ( 9)*16))
            V4FMADDPS(ZMM(18), ZMM(0), MEM(RBX, (10)*16))
            V4FMADDPS(ZMM(19), ZMM(0), MEM(RBX, (11)*16))
            V4FMADDPS(ZMM(20), ZMM(0), MEM(RBX, (12)*16))
            V4FMADDPS(ZMM(21), ZMM(0), MEM(RBX, (13)*16))
            V4FMADDPS(ZMM(22), ZMM(0), MEM(RBX, (14)*16))
            V4FMADDPS(ZMM(23), ZMM(0), MEM(RBX, (15)*16))
            V4FMADDPS(ZMM(24), ZMM(0), MEM(RBX, (16)*16))
            V4FMADDPS(ZMM(25), ZMM(0), MEM(RBX, (17)*16))
            V4FMADDPS(ZMM(26), ZMM(0), MEM(RBX, (18)*16))
            V4FMADDPS(ZMM(27), ZMM(0), MEM(RBX, (19)*16))
            V4FMADDPS(ZMM(28), ZMM(0), MEM(RBX, (20)*16))
            V4FMADDPS(ZMM(29), ZMM(0), MEM(RBX, (21)*16))
            V4FMADDPS(ZMM(30), ZMM(0), MEM(RBX, (22)*16))
            V4FMADDPS(ZMM(31), ZMM(0), MEM(RBX, (23)*16))

            // Update pointers to a and b
            ADD(RAX, IMM(256)) // RAX += 16*4*sizeof(float)
            ADD(RBX, IMM(384)) // RAX += 24*4*sizeof(float)
            
            // Update loop counter and compare
            SUB(RSI, IMM(1))
        JNZ(MAIN_LOOP)


        // Load alpha and beta
        MOV(RAX, VAR(alpha))
        MOV(RBX, VAR(beta))
        VBROADCASTSS(ZMM(0), MEM(RAX))
        VBROADCASTSS(ZMM(1), MEM(RBX))

        // Update C
        UPDATE_C_FOUR_COLS(8,  9,  10, 11) 
        UPDATE_C_FOUR_COLS(12, 13, 14, 15) 
        UPDATE_C_FOUR_COLS(16, 17, 18, 19) 
        UPDATE_C_FOUR_COLS(20, 21, 22, 23) 
        UPDATE_C_FOUR_COLS(24, 25, 26, 27) 
        UPDATE_C_FOUR_COLS(28, 29, 30, 31) 
    : // output operands
    : // input operands
      [k]         "m" (k),
      [a]         "m" (a),
      [b]         "m" (b),
      [alpha]     "m" (alpha),
      [beta]      "m" (beta),
      [c]         "m" (c),
      [rs_c]      "m" (rs_c),
      [cs_c]      "m" (cs_c)
    : // register clobber list
      "rax", "rbx", "rcx", "rsi", "r12", "r13", 
      "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
      "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13",
      "zmm14", "zmm15", "zmm16", "zmm17", "zmm18", "zmm19", "zmm20", "zmm21",
      "zmm22", "zmm23", "zmm24", "zmm25", "zmm26", "zmm27", "zmm28", "zmm29",
      "zmm30", "zmm31", "memory"
    );
}
