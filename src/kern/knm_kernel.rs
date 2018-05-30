use matrix::{Scalar,Mat,RoCM,Matrix};
use core::{ptr, marker::PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::Unsigned;
use super::knm_kernel_wrapper::{KnmKernelWrapper,GenericKnmKernelWrapper};

pub struct KnmKernel<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned>{
    tmp: Matrix<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _mrt: PhantomData<Mr>,
    _nrt: PhantomData<Nr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for KnmKernel<T, At, Bt, Ct, Mr, Nr> {
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) -> () {
        for z in 0..a.width() {
            for y in 0..c.height() {
                for x in 0..c.width() {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set(y, x, t);
                }   
            }   
        }   
    }   
    fn new() -> Self {
        KnmKernel{ tmp: <Matrix<T>>::new(Mr::to_usize(), Nr::to_usize()),
            _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _mrt: PhantomData, _nrt: PhantomData } 
    }   
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::K{bsz: 4});
        desc
    }
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for KnmKernel<T, At, Bt, Ct, Mr, Nr> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) -> () {
        debug_assert!(c.height() <= Mr::to_usize());
        debug_assert!(c.width() <= Nr::to_usize());
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let c_leaf_rs = c.get_leaf_rs() as isize;
        let c_leaf_cs = c.get_leaf_cs() as isize;

        let mut alpha = a.get_scalar() * b.get_scalar();
        let mut beta = c.get_scalar();

        let k = a.width() as isize;

        if Ct::full_leaves() || (c.height() == Mr::to_usize() && c.width() == Nr::to_usize()) {
            <KnmKernelWrapper<Mr,Nr,T>>::run(k, &mut alpha, ap, bp, &mut beta, cp, c_leaf_rs, c_leaf_cs);
        }
        else {
            let tp = self.tmp.get_mut_buffer();
            let t_rs = self.tmp.get_row_stride() as isize;
            let t_cs = self.tmp.get_column_stride() as isize;
            let mut zero = T::zero();

            <KnmKernelWrapper<Mr,Nr,T>>::run(k, &mut alpha, ap, bp, &mut zero, tp, t_rs, t_cs);

            //Add t to c
            for ii in 0..c.height() as isize {
                for jj in 0..c.width() as isize {
                    let tau = ptr::read(tp.offset(ii * t_rs + jj * t_cs));
                    let chi = ptr::read(cp.offset(ii * c_leaf_rs + jj * c_leaf_cs));
                    ptr::write(cp.offset(ii * c_leaf_rs + jj * c_leaf_cs), tau+beta*chi);
                }
            }
        }
    }   
}
