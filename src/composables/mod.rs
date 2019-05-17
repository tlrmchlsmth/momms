//mod gemm;
mod part;
mod pack;
mod parallel_range;
mod spawn;
mod barrier;
mod triple_loop;
mod unpack;
mod fused_pack;

//pub use self::gemm::{GemmNode,AlgorithmStep};
pub use self::part::{PartM,PartN,PartK};
pub use self::pack::{PackA,PackB};
pub use self::parallel_range::{ParallelM,ParallelN,Nwayer,TheRest,Target};
pub use self::spawn::{SpawnThreads};
pub use self::barrier::{Barrier};
pub use self::triple_loop::{TripleLoop};
pub use self::unpack::{UnpackC};
pub use self::fused_pack::{DelayedPackA,DelayedPackB,UnpairA,UnpairB,UnpairC};

use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;

#[derive(Copy, Clone)]
pub enum AlgorithmStep {
    M { bsz: usize },
    N { bsz: usize },
    K { bsz: usize },
}

pub trait GemmNode<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> ();
    fn new() -> Self;
    fn hierarchy_description() -> Vec<AlgorithmStep>;
}
