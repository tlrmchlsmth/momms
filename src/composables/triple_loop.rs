pub use matrix::{Scalar,Mat};
pub use composables::{GemmNode,AlgorithmStep};
pub use thread_comm::ThreadInfo;

pub struct TripleLoop{}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for TripleLoop {
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) -> () {
        for x in 0..c.width() {
            for z in 0..a.width() {
                for y in 0..c.height() {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set(y, x, t);
                }
            }
        }
    }
    fn new() -> Self { TripleLoop{} }
    fn hierarchy_description() -> Vec<AlgorithmStep> { Vec::new() }  
}
