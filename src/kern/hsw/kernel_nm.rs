use matrix::{Scalar,Mat,Hierarch,Matrix,RowPanelMatrix,ColumnPanelMatrix,RoCM};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{Unsigned};

extern crate libc;
use self::libc::{ c_double, int64_t };

type T = f64;
pub struct KernelNM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned>
    //where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
    _nrt: PhantomData<Nr>,
    _mrt: PhantomData<Mr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for KernelNM<T, At, Bt, Ct, Nr, Mr> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
    #[inline(always)]
    default unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        panic!("Macrokernel general case not implemented! for NR {} MR {}", Nr::to_usize(), Mr::to_usize());
        }
    }
    fn new( ) -> KernelNM<T, At, Bt, Ct, Nr, Mr> { 
        KernelNM{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData, _nrt: PhantomData, _mrt: PhantomData } 
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::M{bsz: Mr::to_usize()});
        desc.push(AlgorithmStep::N{bsz: Nr::to_usize()});
        desc
    }  
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for KernelNM<T, At, Bt, Ct, Nr, Mr>
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>,
          Nr: U8, Mr: U6
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) { 
        //A must be column major and B must be row major 
        debug_assert!(a.get_leaf_rs() == 1 && a.get_leaf_cs() == Mr::to_usize());
        debug_assert!(b.get_leaf_cs() == 1 && b.get_leaf_rs() == Nr::to_usize());

        let ap = a.get_buffer();
        let bp = b.get_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_leaf_rs = c.get_leaf_rs() as int64_t;
        let c_leaf_cs = c.get_leaf_cs() as int64_t;

        let c_nr_stride = c.get_block_cs(1, Nr::to_usize()) as isize;
        let c_mr_stride = c.get_block_rs(1, Mr::to_usize()) as isize;

        let mut c_jr = cp;
        let mut jr : isize = 0;
        while jr < n {
            let b_jr = bp.offset(jr * K::to_isize());
    
            let mut ir : isize = 0;
            while ir < m {
                let a_ir = ap.offset(ir * K::to_isize());
                let c_ir = c_jr.offset(ir * Mr::to_isize());

                bli_dgemm_asm_6x8 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_ir as *mut c_double,
                    c_leaf_rs, c_leaf_cs);
                ir += Mr::to_isize();
            }
            jr += Nr::to_isize();
            c_jr = c_jr.offset(c_nr_stride);
    }
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for KernelNM<T, At, Bt, Ct, Nr, Mr>
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>,
          Nr: U12, Mr: U4
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) { 
        debug_assert!(a.get_leaf_rs() == 1 && a.get_leaf_cs() == Mr::to_usize());
        debug_assert!(b.get_leaf_cs() == 1 && b.get_leaf_rs() == Nr::to_usize());

        let ap = a.get_buffer();
        let bp = b.get_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_leaf_rs = c.get_leaf_rs() as int64_t;
        let c_leaf_cs = c.get_leaf_cs() as int64_t;

        let c_nr_stride = c.get_block_cs(1, Nr::to_usize()) as isize;
        let c_mr_stride = c.get_block_rs(1, Mr::to_usize()) as isize;

        let mut c_jr = cp;
        let mut jr : isize = 0;
        while jr < n {
            let b_jr = bp.offset(jr * K::to_isize());
    
            let mut ir : isize = 0;
            while ir < m {
                let a_ir = ap.offset(ir * K::to_isize());
                let c_ir = c_jr.offset(ir * Mr::to_isize());

                bli_dgemm_asm_4x12 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_ir as *mut c_double,
                    c_leaf_rs, c_leaf_cs);
                ir += Mr::to_isize();
            }
            jr += Nr::to_isize();
            c_jr = c_jr.offset(c_nr_stride);
    }
}
