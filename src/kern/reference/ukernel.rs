//use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix,Hierarch};
use matrix::{Scalar,Mat};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};

pub struct Ukernel<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct> {
    #[inline(always)]
    default unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        for z in 0..a.width() {
            for y in 0..c.height() {
                for x in 0..c.width() {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
    }
    fn new( ) -> Ukernel<T, At, Bt, Ct> { 
        Ukernel{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData } 
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        Vec::new()
    }  
}

