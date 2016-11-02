use matrix::{Scalar,Mat,Hierarch};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{Unsigned, U1};

pub struct GemvAL1<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mc: Unsigned, Kc: Unsigned>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
    _mct: PhantomData<Mc>,
    _kct: PhantomData<Kc>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mc: Unsigned, Kc: Unsigned> 
    GemmNode<T, At, Bt, Ct> for 
    GemvAL1<T, At, Bt, Ct, Mc, Kc> {
    #[inline(always)]
    default unsafe fn run( &mut self, _a: &mut At, _b: &mut Bt, _c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        panic!("Macrokernel general case not implemented!");
    }
    fn new( ) -> GemvAL1<T, At, Bt, Ct, Mc, Kc> { 
        GemvAL1{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData, _mct: PhantomData, _kct: PhantomData } 
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::N{bsz: U1::to_usize()});
        desc
    }  
}

use typenum::{U60};
type T = f64;
type Mc = U60;
impl<Kc: Unsigned, N: Unsigned>
     GemmNode<T, Hierarch<T, Mc, Kc, U1, Mc>,
                 Hierarch<T, Kc, N,  U1, Kc>,
                 Hierarch<T, Mc, N,  U1, Mc>> for
     GemvAL1<T, Hierarch<T, Mc, Kc, U1, Mc>,
                Hierarch<T, Kc, N,  U1, Kc>,
                Hierarch<T, Mc, N,  U1, Mc>, Mc, Kc>
{
    #[inline(always)]
    unsafe fn run(&mut self, 
        a: &mut Hierarch<T, Mc, Kc, U1, Mc>,
        b: &mut Hierarch<T, Kc, N,  U1, Kc>, 
        c: &mut Hierarch<T, Mc, N,  U1, Mc>, _thr: &ThreadInfo<T>)
    {
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let k = a.width() as isize;
        debug_assert!(m <= Mc::to_isize() && k <= Kc::to_isize());
        let n = c.width() as isize;

        let mut x: isize = 0;
        let mut c_x = cp;
        let mut b_x = bp;
        while x < n {
            //Clear out C registers
            asm!("vzeroall");
            asm!("
                prefetcht0 0*4($0)
                prefetcht0 8*8($0)
                "
                : : "r"(c_x));

            let mut z: isize = 0;
            let mut a_z = ap;
            let mut b_z = b_x;
            while z < k {
                //Perform an AXPY:

                //Broadcast an element of B into a register
                
                //For every Mc/4 elements of the vector of C:
                //Load element of A
                //Multiply them with the element of B and update corresponding SIMD register of C
                asm!("
                    vbroadcastsd ymm15, [$1 + 0]
                    vfmadd231pd ymm0, ymm15, [$0 + 0]
                    vfmadd231pd ymm1, ymm15, [$0 + 8*4]
                    vfmadd231pd ymm2, ymm15, [$0 + 8*8]
                    vfmadd231pd ymm3, ymm15, [$0 + 8*12]
                    vfmadd231pd ymm4, ymm15, [$0 + 8*16]
                    vfmadd231pd ymm5, ymm15, [$0 + 8*20]
                    vfmadd231pd ymm6, ymm15, [$0 + 8*24]
                    vfmadd231pd ymm7, ymm15, [$0 + 8*28]
                    vfmadd231pd ymm8, ymm15, [$0 + 8*32]
                    vfmadd231pd ymm9, ymm15, [$0 + 8*36]
                    vfmadd231pd ymm10, ymm15, [$0 + 8*40]
                    vfmadd231pd ymm11, ymm15, [$0 + 8*44]
                    vfmadd231pd ymm12, ymm15, [$0 + 8*48]
                    vfmadd231pd ymm13, ymm15, [$0 + 8*52]
                    vfmadd231pd ymm14, ymm15, [$0 + 8*56]
                    "
                    : : "r"(a_z),"r"(b_z) : :"intel");

                z += 1;
                a_z = a_z.offset(Mc::to_isize());
                b_z = b_z.offset(1 as isize);
            }

            //Update C

            asm!("
                    vaddpd ymm0, ymm0, [$0+8*0]
                    vaddpd ymm1, ymm1, [$0+8*4]
                    vaddpd ymm2, ymm2, [$0+8*8]
                    vaddpd ymm3, ymm3, [$0+8*12]
                    vaddpd ymm4, ymm4, [$0+8*16]
                    vaddpd ymm5, ymm5, [$0+8*20]
                    vaddpd ymm6, ymm6, [$0+8*24]
                    vaddpd ymm7, ymm7, [$0+8*28]
                    vaddpd ymm8, ymm8, [$0+8*32]
                    vaddpd ymm9, ymm9, [$0+8*36]
                    vaddpd ymm10, ymm10, [$0+8*40]
                    vaddpd ymm11, ymm11, [$0+8*44]
                    vaddpd ymm12, ymm12, [$0+8*48]
                    vaddpd ymm13, ymm13, [$0+8*52]
                    vaddpd ymm14, ymm14, [$0+8*56]

                    vmovapd [$0+8*0], ymm0
                    vmovapd [$0+8*4], ymm1
                    vmovapd [$0+8*8], ymm2
                    vmovapd [$0+8*12], ymm3
                    vmovapd [$0+8*16], ymm4
                    vmovapd [$0+8*20], ymm5
                    vmovapd [$0+8*24], ymm6
                    vmovapd [$0+8*28], ymm7
                    vmovapd [$0+8*32], ymm8
                    vmovapd [$0+8*36], ymm9
                    vmovapd [$0+8*40], ymm10
                    vmovapd [$0+8*44], ymm11
                    vmovapd [$0+8*48], ymm12
                    vmovapd [$0+8*52], ymm13
                    vmovapd [$0+8*56], ymm14
                "
                : : "r"(c_x) : "memory" :"intel");

            x += 1;
            c_x = c_x.offset(Mc::to_isize());
            b_x = b_x.offset(Kc::to_isize());
        }
    }
}


use typenum::{U56};
impl<Kc: Unsigned, N: Unsigned>
     GemmNode<T, Hierarch<T, U56, Kc, U1, U56>,
                 Hierarch<T, Kc,  N,  U1, Kc>,
                 Hierarch<T, U56, N,  U1, U56>> for
     GemvAL1<T, Hierarch<T, U56, Kc, U1, U56>,
                Hierarch<T, Kc,  N,  U1, Kc>,
                Hierarch<T, U56, N,  U1, U56>, U56, Kc>
{
    #[inline(always)]
    unsafe fn run(&mut self, 
        a: &mut Hierarch<T, U56, Kc, U1, U56>,
        b: &mut Hierarch<T, Kc,  N,  U1, Kc>, 
        c: &mut Hierarch<T, U56, N,  U1, U56>, _thr: &ThreadInfo<T>)
    {
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let k = a.width() as isize;
        let n = c.width() as isize;
        debug_assert!(m <= U56::to_isize() && k <= Kc::to_isize());

        let mut x: isize = 0;
        let mut c_x = cp;
        let mut b_x = bp;
        while x < n {
            //Clear out C registers
            asm!("vzeroall");
            asm!("
                prefetcht0 0*4($0)
                prefetcht0 8*8($0)
                "
                : : "r"(c_x));

            let mut z: isize = 0;
            let mut a_z = ap;
            while z < k {
                //Perform an AXPY:

                //Broadcast an element of B into a register
                
                //For every Mc/4 elements of the vector of C:
                //Load element of A
                //Multiply them with the element of B and update corresponding SIMD register of C
                asm!("
                    vbroadcastsd ymm15, [$1 + 8*$2]
                    vfmadd231pd ymm0, ymm15, [$0 + 0]
                    vfmadd231pd ymm1, ymm15, [$0 + 8*4]
                    vfmadd231pd ymm2, ymm15, [$0 + 8*8]
                    vfmadd231pd ymm3, ymm15, [$0 + 8*12]
                    vfmadd231pd ymm4, ymm15, [$0 + 8*16]
                    vfmadd231pd ymm5, ymm15, [$0 + 8*20]
                    vfmadd231pd ymm6, ymm15, [$0 + 8*24]
                    vfmadd231pd ymm7, ymm15, [$0 + 8*28]
                    vfmadd231pd ymm8, ymm15, [$0 + 8*32]
                    vfmadd231pd ymm9, ymm15, [$0 + 8*36]
                    vfmadd231pd ymm10, ymm15, [$0 + 8*40]
                    vfmadd231pd ymm11, ymm15, [$0 + 8*44]
                    vfmadd231pd ymm12, ymm15, [$0 + 8*48]
                    vfmadd231pd ymm13, ymm15, [$0 + 8*52]
                    "
                    : : "r"(a_z),"r"(b_x), "r"(z) : :"intel");

                z += 4;
                a_z = a_z.offset(U56::to_isize());
            }

            //Update C
            asm!("
                    vaddpd ymm0, ymm0, [$0+8*0]
                    vaddpd ymm1, ymm1, [$0+8*4]
                    vaddpd ymm2, ymm2, [$0+8*8]
                    vaddpd ymm3, ymm3, [$0+8*12]
                    vaddpd ymm4, ymm4, [$0+8*16]
                    vaddpd ymm5, ymm5, [$0+8*20]
                    vaddpd ymm6, ymm6, [$0+8*24]
                    vaddpd ymm7, ymm7, [$0+8*28]
                    vaddpd ymm8, ymm8, [$0+8*32]
                    vaddpd ymm9, ymm9, [$0+8*36]
                    vaddpd ymm10, ymm10, [$0+8*40]
                    vaddpd ymm11, ymm11, [$0+8*44]
                    vaddpd ymm12, ymm12, [$0+8*48]
                    vaddpd ymm13, ymm13, [$0+8*52]

                    vmovapd [$0+8*0], ymm0
                    vmovapd [$0+8*4], ymm1
                    vmovapd [$0+8*8], ymm2
                    vmovapd [$0+8*12], ymm3
                    vmovapd [$0+8*16], ymm4
                    vmovapd [$0+8*20], ymm5
                    vmovapd [$0+8*24], ymm6
                    vmovapd [$0+8*28], ymm7
                    vmovapd [$0+8*32], ymm8
                    vmovapd [$0+8*36], ymm9
                    vmovapd [$0+8*40], ymm10
                    vmovapd [$0+8*44], ymm11
                    vmovapd [$0+8*48], ymm12
                    vmovapd [$0+8*52], ymm13
                "
                : : "r"(c_x) : "memory" :"intel");

            x += 1;
            c_x = c_x.offset(U56::to_isize());
            b_x = b_x.offset(Kc::to_isize());
        }
    }
}
