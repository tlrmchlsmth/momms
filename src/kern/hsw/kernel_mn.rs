use matrix::{Scalar,Mat,Hierarch};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{U1,U4,U6,U8,U12,Unsigned};

pub struct KernelMN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
    _nrt: PhantomData<Nr>,
    _mrt: PhantomData<Mr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for 
    KernelMN<T, At, Bt, Ct, Mr, Nr> {
    #[inline(always)]
    default unsafe fn run( &mut self, _a: &mut At, _b: &mut Bt, _c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        panic!("Macrokernel general case not implemented!");
    }
    fn new( ) -> KernelMN<T, At, Bt, Ct, Mr, Nr> { 
        KernelMN{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData, _nrt: PhantomData, _mrt: PhantomData } 
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::N{bsz: Nr::to_usize()});
        desc.push(AlgorithmStep::M{bsz: Mr::to_usize()});
        desc
    }  
}

type T = f64;
impl<K: Unsigned>
    GemmNode<T, Hierarch<T, U4, K,  U1, U4>,
                Hierarch<T, K,  U12, U12, U1>,
                Hierarch<T, U4, U12, U12, U1>> for
    KernelMN<T, Hierarch<T, U4, K,  U1, U4>,
                Hierarch<T, K,  U12, U12, U1>,
                Hierarch<T, U4, U12, U12, U1>, U4, U12> 
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Hierarch<T, U4, K, U1, U4>,
        b: &mut Hierarch<T, K, U12, U12, U1>, 
        c: &mut Hierarch<T, U4, U12, U12, U1>, _thr: &ThreadInfo<T>)
    {
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let c_mr_stride = c.block_stride_y(0) as isize;

        let mut ir : isize = 0;
        let mut c_ir = cp;
        while ir < m {
            let a_ir = ap.offset(ir * K::to_isize());

            let mut jr : isize = 0;
            while jr < n {
                let b_jr = bp.offset(jr * K::to_isize());
                let c_jr = c_ir.offset(jr * U4::to_isize());

                asm!("vzeroall");
                asm!("
                    prefetcht0 0*8($0)
                    prefetcht0 24*8($0)
                    "
                    : : "r"(c_jr));
                /*asm!("
                    prefetcht0 0*8($0)
                    prefetcht0 8*8($0)
                    prefetcht0 16*8($0)
                    prefetcht0 24*8($0)
                    prefetcht0 32*8($0)
                    prefetcht0 40*8($0)
                    "
                    : : "r"(cp));*/
                //ymm12: a
                //ymm13: b1
                //ymm14: b2
                //ymm15: b3
                let mut a_ind : isize = 0;
                let mut b_ind : isize = 0;
                for _ in 0..k{
                    asm!("
                        vmovapd ymm13, [$3 + 8*$1 + 0]
                        vmovapd ymm14, [$3 + 8*$1 + 32]
                        vmovapd ymm15, [$3 + 8*$1 + 64]

                        vmovapd ymm12, [$2 + 8*$0 + 0]
                        vfmadd231pd ymm0, ymm12, ymm13
                        vfmadd231pd ymm1, ymm12, ymm14
                        prefetcht0 [$2+$0+8*8]
                        vfmadd231pd ymm2, ymm12, ymm15
            
                        vpermpd ymm12, ymm12, 0xB1
                        vfmadd231pd ymm3, ymm12, ymm13
                        vfmadd231pd ymm4, ymm12, ymm14
                        vfmadd231pd ymm5, ymm12, ymm15
                        
                        vperm2f128 ymm12, ymm12, ymm12, 1
                        vfmadd231pd ymm6, ymm12, ymm13
                        vfmadd231pd ymm7, ymm12, ymm14
                        vfmadd231pd ymm8, ymm12, ymm15

                        vpermpd ymm12, ymm12, 0xB1
                        vfmadd231pd ymm9, ymm12, ymm13
                        vfmadd231pd ymm10, ymm12, ymm14
                        vfmadd231pd ymm11, ymm12, ymm15
                        " 
                        : : "r"(a_ind),"r"(b_ind),"r"(a_ir),"r"(b_jr) : "ymm12 ymm13 ymm14 ymm15" :"intel");
                        a_ind += 4;
                        b_ind += 12;

                }
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
                "
                : : "r"(c_jr) : "memory" :"intel");

                jr += U12::to_isize();
            }
            ir += U4::to_isize();
            c_ir = c_ir.offset(c_mr_stride);
        }
    }
}

extern crate libc;
use self::libc::{ c_double, int64_t };

extern{
    fn bli_dgemm_asm_6x8 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}
impl<K: Unsigned>
    GemmNode<T, Hierarch<T, U6, K,  U1, U6>,
                Hierarch<T, K,  U8, U8, U1>,
                Hierarch<T, U6, U8, U8, U1>> for
    KernelMN<T, Hierarch<T, U6, K,  U1, U6>,
                Hierarch<T, K,  U8, U8, U1>,
                Hierarch<T, U6, U8, U8, U1>, U6, U8> 
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Hierarch<T, U6, K, U1, U6>,
        b: &mut Hierarch<T, K, U8, U8, U1>, 
        c: &mut Hierarch<T, U6, U8, U8, U1>, _thr: &ThreadInfo<T>)
    {
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_mr_stride = c.block_stride_y(0) as isize;

        let mut ir : isize = 0;
        let mut c_ir = cp;
        while ir < m {
            let a_ir = ap.offset(ir * K::to_isize());

            let mut jr : isize = 0;
            while jr < n {
                let b_jr = bp.offset(jr * K::to_isize());
                let c_jr = c_ir.offset(jr * U6::to_isize());

                bli_dgemm_asm_6x8 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_jr as *mut c_double,
                    U8::to_isize() as int64_t, U1::to_isize() as int64_t );

                jr += U8::to_isize();
            }
            ir += U6::to_isize();
            c_ir = c_ir.offset(c_mr_stride);
        }
    }
}
