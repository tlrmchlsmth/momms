use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use core::marker::PhantomData;

pub struct Barrier<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}

impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for Barrier<T,At,Bt,Ct,S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T> ) -> () {
        thr.barrier();
        self.child.run(a, b, c, thr);
    }
    fn new( ) -> Barrier<T,At,Bt,Ct,S>{
            Barrier{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}
