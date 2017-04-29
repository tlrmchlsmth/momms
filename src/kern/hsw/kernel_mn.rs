use matrix::{Scalar,Mat,RoCM};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{U4,U6,U8,U12,Unsigned};

extern crate libc;
use self::libc::{ c_double, int64_t };
extern{
    fn bli_dgemm_asm_6x8 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
    fn bli_dgemm_asm_4x12 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
    fn bli_dgemm_asm_12x4 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}

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

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for KernelMN<T, At, Bt, Ct, U6, U8>
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>,
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) { 
        //A must be column major and B must be row major 
        debug_assert!(a.get_leaf_rs() == 1 && a.get_leaf_cs() == U6::to_usize());
        debug_assert!(b.get_leaf_cs() == 1 && b.get_leaf_rs() == U8::to_usize());

        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_leaf_rs = c.get_leaf_rs() as int64_t;
        let c_leaf_cs = c.get_leaf_cs() as int64_t;

        let c_nr_stride = c.get_block_cs(1, U8::to_usize()) as isize;
        let b_nr_stride = b.get_block_cs(1, U8::to_usize()) as isize;

        let c_mr_stride = c.get_block_rs(1, U6::to_usize()) as isize;
        let a_mr_stride = a.get_block_rs(1, U6::to_usize()) as isize;

        let mut ir : isize = 0;
        let mut a_ir = ap;
        let mut c_ir = cp;
        while ir < m {
    
            let mut c_jr = c_ir;
            let mut b_jr = bp;
            let mut jr : isize = 0;
            while jr < n {
                bli_dgemm_asm_6x8 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_ir as *mut c_double,
                    c_leaf_rs, c_leaf_cs);

                jr += U8::to_isize();
                c_jr = c_jr.offset(c_nr_stride);
                b_jr = b_jr.offset(b_nr_stride);
            }
            ir += U6::to_isize();
            a_ir = a_ir.offset(a_mr_stride);
            c_ir = c_ir.offset(c_mr_stride);
        }
    }
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for KernelMN<T, At, Bt, Ct, U12, U4>
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>,
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) { 
        //A must be column major and B must be row major 
        debug_assert!(b.get_leaf_cs() == 1 && b.get_leaf_rs() == U4::to_usize());
        debug_assert!(a.get_leaf_rs() == 1 && a.get_leaf_cs() == U12::to_usize());

        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_leaf_rs = c.get_leaf_rs() as int64_t;
        let c_leaf_cs = c.get_leaf_cs() as int64_t;

        let c_nr_stride = c.get_block_cs(1, U4::to_usize()) as isize;
        let b_nr_stride = b.get_block_cs(1, U4::to_usize()) as isize;

        let c_mr_stride = c.get_block_rs(1, U12::to_usize()) as isize;
        let a_mr_stride = a.get_block_rs(1, U12::to_usize()) as isize;

        let mut ir : isize = 0;
        let mut a_ir = ap;
        let mut c_ir = cp;
        while ir < m {
    
            let mut c_jr = c_ir;
            let mut b_jr = bp;
            let mut jr : isize = 0;
            while jr < n {
                bli_dgemm_asm_12x4 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_ir as *mut c_double,
                    c_leaf_rs, c_leaf_cs);

                jr += U4::to_isize();
                c_jr = c_jr.offset(c_nr_stride);
                b_jr = b_jr.offset(b_nr_stride);
            }
            ir += U12::to_isize();
            a_ir = a_ir.offset(a_mr_stride);
            c_ir = c_ir.offset(c_mr_stride);
        }
    }
}
