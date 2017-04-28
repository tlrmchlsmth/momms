use matrix::{Scalar,Mat,RoCM};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::Unsigned;
use super::ukernel_wrapper::{UkernelWrapper,GenericUkernelWrapper};

pub struct KernelMN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
    _mrt: PhantomData<Nr>,
    _nrt: PhantomData<Mr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for 
    KernelMN<T, At, Bt, Ct, Mr, Nr> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T> {
    #[inline(always)]
    default unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        //A must be column major and B must be row major 
        debug_assert!(a.get_leaf_rs() == 1 && a.get_leaf_cs() == Mr::to_usize());
        debug_assert!(b.get_leaf_cs() == 1 && b.get_leaf_rs() == Nr::to_usize());

        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha = a.get_scalar() * b.get_scalar();
        let mut beta = c.get_scalar();

        let c_leaf_rs = c.get_leaf_rs() as isize;
        let c_leaf_cs = c.get_leaf_cs() as isize;

        let c_mr_stride = c.get_block_rs(1, Mr::to_usize()) as isize;
        let a_mr_stride = a.get_block_rs(1, Mr::to_usize()) as isize;

        let c_nr_stride = c.get_block_cs(1, Nr::to_usize()) as isize;
        let b_nr_stride = b.get_block_cs(1, Nr::to_usize()) as isize;

        let mut ir : isize = 0;
        let mut a_ir = ap;
        let mut c_ir = cp;
        while ir < m {
    
            let mut c_jr = c_ir;
            let mut b_jr = bp;
            let mut jr : isize = 0;
            while jr < n {
                <UkernelWrapper<Mr, Nr, T>>::run( k, &mut alpha, a_ir, b_jr, &mut beta, c_jr, c_leaf_rs, c_leaf_cs);

                jr += Nr::to_isize();
                c_jr = c_jr.offset(c_nr_stride);
                b_jr = b_jr.offset(b_nr_stride);
            }
            ir += Mr::to_isize();
            a_ir = a_ir.offset(a_mr_stride);
            c_ir = c_ir.offset(c_mr_stride);
        }
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
