use matrix::{Scalar,Mat,Hierarch};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{U1,U4,U6,U8,U12,Unsigned};

type T = f64;

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
