use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;

pub trait GemmNode<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T> ) -> ();
//    #[inline(always)]
//    unsafe fn shadow( &self ) -> Self where Self: Sized;
    fn new( ) -> Self;
}
