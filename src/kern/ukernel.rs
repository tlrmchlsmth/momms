use matrix::{Scalar,Mat,RoCM,Matrix};
use core::ptr;
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use super::ukernel_wrapper::{UkernelWrapper,GenericUkernelWrapper};

pub struct Ukernel<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, const Mr: usize, const Nr: usize>{
    tmp: Matrix<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, const Mr: usize, const Nr: usize> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct, {Mr}, {Nr}> {
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
        let mut tmp = <Matrix<T>>::new(Nr, Mr);
        tmp.transpose();
        Ukernel{ tmp: tmp, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }   
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        Vec::new()
    }   
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, const Mr: usize, const Nr: usize> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct, {Mr}, {Nr}> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) -> () {
        debug_assert!(c.height() <= Mr);
        debug_assert!(c.width() <= Nr);
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let c_leaf_rs = c.get_leaf_rs() as isize;
        let c_leaf_cs = c.get_leaf_cs() as isize;

        let mut alpha = a.get_scalar() * b.get_scalar();
        let mut beta = c.get_scalar();

        let k = a.width() as isize;

        if Ct::full_leaves() || (c.height() == Mr && c.width() == Nr) {
            <UkernelWrapper<T,{Mr},{Nr}>>::run(k, &mut alpha, ap, bp, &mut beta, cp, c_leaf_rs, c_leaf_cs);
        }
        else {
            let tp = self.tmp.get_mut_buffer();
            let t_rs = self.tmp.get_row_stride() as isize;
            let t_cs = self.tmp.get_column_stride() as isize;
            let mut zero = T::zero();
            <UkernelWrapper<T,{Mr},{Nr}>>::run(k, &mut alpha, ap, bp, &mut zero, tp, t_rs, t_cs);

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
