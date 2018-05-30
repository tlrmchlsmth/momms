#include "blis.h"
#include <immintrin.h>

/*
 * Current panel of A: Column-major
 * Current panel of B: Array of blocks. There are k/4 blocks. Each block is a contiguous column-major, 4 by n_r
 *  (For a less efficient implementation, B could be column-major)
 * Current block of C: Column-major 
 */
#define NR 24

void sgemm_knm_int_16x24
     (
       dim_t               k_,
       float*    restrict alpha_,
       float*    restrict a,
       float*    restrict b,
       float*    restrict beta_,
       float*    restrict c, inc_t rs_c, inc_t cs_c
     )
{
    int64_t k = k_;

    //
    // Zero out accumulators for C
    //
    __m512 c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24;
    c1 = _mm512_setzero_ps();
    c2 = _mm512_setzero_ps();
    c3 = _mm512_setzero_ps();
    c4 = _mm512_setzero_ps();
    c5 = _mm512_setzero_ps();
    c6 = _mm512_setzero_ps();
    c7 = _mm512_setzero_ps();
    c8 = _mm512_setzero_ps();
    c9 = _mm512_setzero_ps();
    c10 = _mm512_setzero_ps();
    c11 = _mm512_setzero_ps();
    c12 = _mm512_setzero_ps();
    c13 = _mm512_setzero_ps();
    c14 = _mm512_setzero_ps();
    c15 = _mm512_setzero_ps();
    c16 = _mm512_setzero_ps();
    c17 = _mm512_setzero_ps();
    c18 = _mm512_setzero_ps();
    c19 = _mm512_setzero_ps();
    c20 = _mm512_setzero_ps();
    c21 = _mm512_setzero_ps();
    c22 = _mm512_setzero_ps();
    c23 = _mm512_setzero_ps();
    c24 = _mm512_setzero_ps();

    //
    // Main Loop
    //
    for (int p = 0; p < k; p += 4) {
        //
        // Load 4 vectors of A
        //
        __m512 a1 = _mm512_load_ps(a + 16 * (p+0));
        __m512 a2 = _mm512_load_ps(a + 16 * (p+1));
        __m512 a3 = _mm512_load_ps(a + 16 * (p+2));
        __m512 a4 = _mm512_load_ps(a + 16 * (p+3));

        //
        // Perform NR: 16x4 MVMs
        c1 = _mm512_4fmadd_ps(c1, a1, a2, a3, a4, b + NR*p + 0);
        c2 = _mm512_4fmadd_ps(c2, a1, a2, a3, a4, b + NR*p + 4);
        c3 = _mm512_4fmadd_ps(c3, a1, a2, a3, a4, b + NR*p + 8);
        c4 = _mm512_4fmadd_ps(c4, a1, a2, a3, a4, b + NR*p + 12);
        c5 = _mm512_4fmadd_ps(c5, a1, a2, a3, a4, b + NR*p + 16);
        c6 = _mm512_4fmadd_ps(c6, a1, a2, a3, a4, b + NR*p + 20);
        c7 = _mm512_4fmadd_ps(c7, a1, a2, a3, a4, b + NR*p + 24);
        c8 = _mm512_4fmadd_ps(c8, a1, a2, a3, a4, b + NR*p + 28);
        c9 = _mm512_4fmadd_ps(c9, a1, a2, a3, a4, b + NR*p + 32);
        c10 = _mm512_4fmadd_ps(c10, a1, a2, a3, a4, b + NR*p + 36);
        c11 = _mm512_4fmadd_ps(c11, a1, a2, a3, a4, b + NR*p + 40);
        c12 = _mm512_4fmadd_ps(c12, a1, a2, a3, a4, b + NR*p + 44);
        c13 = _mm512_4fmadd_ps(c13, a1, a2, a3, a4, b + NR*p + 48);
        c14 = _mm512_4fmadd_ps(c14, a1, a2, a3, a4, b + NR*p + 52);
        c15 = _mm512_4fmadd_ps(c15, a1, a2, a3, a4, b + NR*p + 56);
        c16 = _mm512_4fmadd_ps(c16, a1, a2, a3, a4, b + NR*p + 60);
        c17 = _mm512_4fmadd_ps(c17, a1, a2, a3, a4, b + NR*p + 64);
        c18 = _mm512_4fmadd_ps(c18, a1, a2, a3, a4, b + NR*p + 68);
        c19 = _mm512_4fmadd_ps(c19, a1, a2, a3, a4, b + NR*p + 72);
        c20 = _mm512_4fmadd_ps(c20, a1, a2, a3, a4, b + NR*p + 76);
        c21 = _mm512_4fmadd_ps(c21, a1, a2, a3, a4, b + NR*p + 80);
        c22 = _mm512_4fmadd_ps(c22, a1, a2, a3, a4, b + NR*p + 84);
        c23 = _mm512_4fmadd_ps(c23, a1, a2, a3, a4, b + NR*p + 88);
        c24 = _mm512_4fmadd_ps(c24, a1, a2, a3, a4, b + NR*p + 92);
    }

    //
    // Load alpha and beta
    //
    __m512 alpha = _mm512_set1_ps(*alpha_);
    __m512 beta = _mm512_set1_ps(*beta_);

    //
    // Update C
    //
    __m512 t;
    t = _mm512_load_ps(c + 0*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c1, t);
    _mm512_store_ps(c + 0*cs_c, t);

    t = _mm512_load_ps(c + 1*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c2, t);
    _mm512_store_ps(c + 1*cs_c, t);

    t = _mm512_load_ps(c + 2*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c3, t);
    _mm512_store_ps(c + 2*cs_c, t);
    
    t = _mm512_load_ps(c + 3*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c4, t);
    _mm512_store_ps(c + 3*cs_c, t);
    
    t = _mm512_load_ps(c + 4*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c5, t);
    _mm512_store_ps(c + 4*cs_c, t);
    
    t = _mm512_load_ps(c + 5*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c6, t);
    _mm512_store_ps(c + 5*cs_c, t);
    
    t = _mm512_load_ps(c + 6*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c7, t);
    _mm512_store_ps(c + 6*cs_c, t);
    
    t = _mm512_load_ps(c + 7*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c8, t);
    _mm512_store_ps(c + 7*cs_c, t);
    
    t = _mm512_load_ps(c + 8*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c9, t);
    _mm512_store_ps(c + 8*cs_c, t);
    
    t = _mm512_load_ps(c + 9*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c10, t);
    _mm512_store_ps(c + 9*cs_c, t);
    
    t = _mm512_load_ps(c + 10*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c11, t);
    _mm512_store_ps(c + 10*cs_c, t);
    
    t = _mm512_load_ps(c + 11*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c12, t);
    _mm512_store_ps(c + 11*cs_c, t);
    
    t = _mm512_load_ps(c + 12*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c13, t);
    _mm512_store_ps(c + 12*cs_c, t);

    t = _mm512_load_ps(c + 13*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c14, t);
    _mm512_store_ps(c + 13*cs_c, t);

    t = _mm512_load_ps(c + 14*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c15, t);
    _mm512_store_ps(c + 14*cs_c, t);

    t = _mm512_load_ps(c + 15*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c16, t);
    _mm512_store_ps(c + 15*cs_c, t);

    t = _mm512_load_ps(c + 16*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c17, t);
    _mm512_store_ps(c + 16*cs_c, t);

    t = _mm512_load_ps(c + 17*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c18, t);
    _mm512_store_ps(c + 17*cs_c, t);

    t = _mm512_load_ps(c + 18*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c19, t);
    _mm512_store_ps(c + 18*cs_c, t);

    t = _mm512_load_ps(c + 19*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c20, t);
    _mm512_store_ps(c + 19*cs_c, t);

    t = _mm512_load_ps(c + 20*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c21, t);
    _mm512_store_ps(c + 20*cs_c, t);

    t = _mm512_load_ps(c + 21*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c22, t);
    _mm512_store_ps(c + 21*cs_c, t);

    t = _mm512_load_ps(c + 22*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c23, t);
    _mm512_store_ps(c + 22*cs_c, t);

    t = _mm512_load_ps(c + 23*cs_c);
    t = _mm512_mul_ps(t, beta);
    t = _mm512_fmadd_ps(alpha, c24, t);
    _mm512_store_ps(c + 23*cs_c, t);
}
