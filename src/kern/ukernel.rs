use matrix::{Scalar,Mat,RoCM,Matrix};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::Unsigned;
use super::ukernel_wrapper::{UkernelWrapper,GenericUkernelWrapper};

pub struct Ukernel<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
    _mrt: PhantomData<Mr>,
    _nrt: PhantomData<Nr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct, Mr, Nr> {
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
    fn new() -> Ukernel<T, At, Bt, Ct, Mr, Nr> { 
        Ukernel{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData, _mrt: PhantomData, _nrt: PhantomData } 
    }   
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        Vec::new()
    }   
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct, Mr, Nr> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T>
{
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

        if c.height() == Mr::to_usize() && c.width() == Nr::to_usize() {
            <UkernelWrapper<Mr,Nr,T>>::run(k, &mut alpha, ap, bp, &mut beta, cp, c_leaf_rs, c_leaf_cs);
        }
        else {
            let mut t : Matrix<T> = Matrix::new(Mr::to_usize(), Nr::to_usize());
            let tp = t.get_mut_buffer();
            let rs_t = t.get_row_stride() as isize;
            let cs_t = t.get_column_stride() as isize;

            let mut zero = T::zero();
            <UkernelWrapper<Mr,Nr,T>>::run(k, &mut alpha, ap, bp, &mut zero, tp, rs_t, cs_t);

            //Copy t to c
            t.push_y_view(c.height());
            t.push_x_view(c.width());
            c.axpby_small(T::one(), &t, beta);
        }
    }   
}
