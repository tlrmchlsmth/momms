use matrix::{Scalar,Mat,RoCM};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::Unsigned;
use super::xsmm_wrapper::*;

pub struct Xsmm<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> {
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>>
    GemmNode<T, At, Bt, Ct> for Xsmm<T, At, Bt, Ct> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) -> () {
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha = a.get_scalar() * b.get_scalar();
        let mut beta = c.get_scalar();
        
        let a_leaf_rs = a.get_leaf_rs() as isize;
        let a_leaf_cs = a.get_leaf_cs() as isize;
        let b_leaf_rs = b.get_leaf_rs() as isize;
        let b_leaf_cs = b.get_leaf_cs() as isize;
        let c_leaf_rs = c.get_leaf_rs() as isize;
        let c_leaf_cs = c.get_leaf_cs() as isize;

        if a_leaf_rs == 1 && b_leaf_rs == 1 && c_leaf_rs == 1 {
            <XsmmWrapper<T>>::run(m,n,k, &mut alpha, ap, a_leaf_cs, bp, b_leaf_cs, &mut beta, cp, c_leaf_cs);
        } else if a_leaf_cs == 1 && b_leaf_cs == 1 && c_leaf_cs == 1 {
            <XsmmWrapper<T>>::run(n,m,k, &mut alpha, bp, b_leaf_rs, ap, a_leaf_rs, &mut beta, cp, c_leaf_rs);
        } else {
            panic!("A, B, and C must all be column major or row major for libxsmm");
        }
    }
    fn new() -> Self { 
        Xsmm{ _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        Vec::new()
    }
}

pub struct KernelXsmmA2<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> {
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _nrt: PhantomData<Nr>,
    _mrt: PhantomData<Mr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for KernelXsmmA2<T, At, Bt, Ct, Nr, Mr> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) -> () {
        //A, B, and C must be row major
        debug_assert!(a.get_leaf_cs() == 1 && b.get_leaf_cs() == 1 && c.get_leaf_cs() == 1);

        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha = a.get_scalar() * b.get_scalar();
        let mut beta = c.get_scalar();
        
        let a_leaf_rs = a.get_leaf_rs() as isize;
        let b_leaf_rs = b.get_leaf_rs() as isize;
        let c_leaf_rs = c.get_leaf_rs() as isize;

        let c_nr_stride = c.get_block_cs(1, Nr::to_usize()) as isize;
        let b_nr_stride = b.get_block_cs(1, Nr::to_usize()) as isize;

        let c_mr_stride = c.get_block_rs(1, Mr::to_usize()) as isize;
        let a_mr_stride = a.get_block_rs(1, Mr::to_usize()) as isize;

        let mut c_jr = cp;
        let mut b_jr = bp;
        let mut jr : isize = 0;
        while jr < n {
            b.establish_leaf(0, (jr as usize) / Nr::to_usize(), k as usize, Nr::to_usize());
            let mut ir : isize = 0;
            let mut a_ir = ap;
            let mut c_ir = c_jr;
            while ir < m {
                //prefetch next C
                let next_c_ir = c_ir.offset(c_mr_stride);
                if cfg!(feature="asm_snippets") {
                    asm!(" prefetcht2 ($0)
                           prefetcht2 64($0)" : : "r"(next_c_ir));
                    asm!(" prefetcht2 ($0)
                           prefetcht2 64($0)" : : "r"(next_c_ir.offset(c_leaf_rs)));
                    asm!(" prefetcht2 ($0)
                           prefetcht2 64($0)" : : "r"(next_c_ir.offset(2*c_leaf_rs)));
                    asm!(" prefetcht2 ($0)
                           prefetcht2 64($0)" : : "r"(next_c_ir.offset(3*c_leaf_rs)));
                }

                let u_m = if m-ir >= Mr::to_isize() { Mr::to_isize() } else { m-ir };
                let u_n = if n-jr >= Nr::to_isize() { Nr::to_isize() } else { n-jr };

                //Call libxsmm to perform
                // C^T += B^T A^T
			    <XsmmWrapper<T>>::run(u_n,u_m,k, &mut alpha, b_jr, b_leaf_rs, a_ir, a_leaf_rs, &mut beta, c_ir, c_leaf_rs);

                ir += Mr::to_isize();
                a_ir = a_ir.offset(a_mr_stride);
                c_ir = c_ir.offset(c_mr_stride);
            }
            jr += Nr::to_isize();
            c_jr = c_jr.offset(c_nr_stride);
            b_jr = b_jr.offset(b_nr_stride);
        }
    }
    fn new() -> Self { 
        KernelXsmmA2{ _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _nrt: PhantomData, _mrt: PhantomData } 
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::M{bsz: Mr::to_usize()});
        desc.push(AlgorithmStep::N{bsz: Nr::to_usize()});
        desc
    }  
}
