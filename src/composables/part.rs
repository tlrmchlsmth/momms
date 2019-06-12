#![feature(const_generics)]

use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use core::marker::PhantomData;

pub struct PartM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>, const NB: usize> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>, const NB: usize>
    GemmNode<T, At, Bt, Ct> for PartM<T,At,Bt,Ct,S,{NB}> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let m_save = a.push_y_view(NB);
        c.push_y_view(NB);
        
        let mut i = 0;
        while i < m_save  {
            a.slide_y_view_to(i, NB);
            c.slide_y_view_to(i, NB);
            
            self.child.run(a, b, c, thr);
            i += NB;
        }

        a.pop_y_view();
        c.pop_y_view();
    }
    fn new() -> Self{
        PartM{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push(AlgorithmStep::M{ bsz: NB });
        child_desc
    }
}

pub struct PartN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>, const NB: usize> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>, const NB: usize>
    GemmNode<T, At, Bt, Ct> for PartN<T,At,Bt,Ct,S,{NB}> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr:&ThreadInfo<T>) -> () {
        let n_save = b.push_x_view(NB);
        c.push_x_view(NB);
        
        let mut i = 0;
        while i < n_save {
            b.slide_x_view_to(i, NB);
            c.slide_x_view_to(i, NB);

            self.child.run(a, b, c, thr);
            i += NB;
        }
        
        b.pop_x_view();
        c.pop_x_view();
    }
    fn new() -> Self {
        PartN{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push(AlgorithmStep::N{ bsz: NB });
        child_desc
    }
}

pub struct PartK<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>, const NB: usize> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>, const NB: usize>
    GemmNode<T, At, Bt, Ct> for PartK<T,At,Bt,Ct,S,{NB}> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let k_save = a.push_x_view(NB);
        b.push_y_view(NB);

        let beta_save = c.get_scalar();

        let mut i = 0;
        while i < k_save  {
            a.slide_x_view_to(i, NB);
            b.slide_y_view_to(i, NB);

            self.child.run(a, b, c, thr);
            i += NB;
            c.set_scalar(T::one());
        }

        a.pop_x_view();
        b.pop_y_view();
        c.set_scalar(beta_save);
    }
    fn new() -> Self {
        PartK{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push(AlgorithmStep::K{ bsz: NB });
        child_desc
    }
}
