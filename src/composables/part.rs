use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use core::marker::PhantomData;
use typenum::Unsigned;

pub struct PartM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartM<T,At,Bt,Ct,Bsz,S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let m_save = a.push_y_view(Bsz::to_usize());
        c.push_y_view(Bsz::to_usize());
        
        let mut i = 0;
        while i < m_save  {
            a.slide_y_view_to(i, Bsz::to_usize());
            c.slide_y_view_to(i, Bsz::to_usize());
            
            self.child.run(a, b, c, thr);
            i += Bsz::to_usize();
        }

        a.pop_y_view();
        c.pop_y_view();
    }
    fn new() -> Self {
            PartM{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push(AlgorithmStep::M{ bsz: Bsz::to_usize() });
        child_desc
    }
}

pub struct PartN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartN<T,At,Bt,Ct,Bsz,S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr:&ThreadInfo<T>) -> () {
        let n_save = b.push_x_view(Bsz::to_usize());
        c.push_x_view(Bsz::to_usize());
        
        let mut i = 0;
        while i < n_save {
            b.slide_x_view_to(i, Bsz::to_usize());
            c.slide_x_view_to(i, Bsz::to_usize());

            self.child.run(a, b, c, thr);
            i += Bsz::to_usize();
        }
        
        b.pop_x_view();
        c.pop_x_view();
    }
    fn new() -> Self {
            PartN{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push(AlgorithmStep::N{ bsz: Bsz::to_usize() });
        child_desc
    }
}

pub struct PartK<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartK<T,At,Bt,Ct,Bsz,S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let k_save = a.push_x_view(Bsz::to_usize());
        b.push_y_view(Bsz::to_usize());

        let beta_save = c.get_scalar();

        let mut i = 0;
        while i < k_save  {
            a.slide_x_view_to(i, Bsz::to_usize());
            b.slide_y_view_to(i, Bsz::to_usize());

            self.child.run(a, b, c, thr);
            i += Bsz::to_usize();
            c.set_scalar(T::one());
        }

        a.pop_x_view();
        b.pop_y_view();
        c.set_scalar(beta_save);
    }
    fn new() -> Self {
        PartK{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S::hierarchy_description();
        child_desc.push(AlgorithmStep::K{ bsz: Bsz::to_usize() });
        child_desc
    }
}

pub struct FirstDiffPartM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, 
    S1: GemmNode<T, At, Bt, Ct>, S2: GemmNode<T, At, Bt, Ct>> {
    child1: S1,
    child2: S2,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S1, S2>
    GemmNode<T, At, Bt, Ct> for FirstDiffPartM<T, At, Bt, Ct, Bsz, S1, S2>
    where S1: GemmNode<T, At, Bt, Ct>,
          S2: GemmNode<T, At, Bt, Ct> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let m_save = a.push_y_view(Bsz::to_usize());
        c.push_y_view(Bsz::to_usize());
        
        let mut i = 0;
        while i < m_save  {
            a.slide_y_view_to(i, Bsz::to_usize());
            c.slide_y_view_to(i, Bsz::to_usize());
            
            if i == 0 { 
                self.child1.run(a, b, c, thr);
            } else {
                self.child2.run(a, b, c, thr);
            }

            i += Bsz::to_usize();
        }

        a.pop_y_view();
        c.pop_y_view();
    }
    fn new() -> Self {
            FirstDiffPartM{ child1: S1::new(), child2: S2::new(),  _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        //There could be two different hierarchy descriptions here!
        //TODO: We might want a runtime check to make sure the two hierarchy descriptions are
        //identical. Maybe only in debug mode.
        let mut child_desc = S1::hierarchy_description();
        child_desc.push(AlgorithmStep::M{ bsz: Bsz::to_usize() });
        child_desc
    }
}

pub struct FirstDiffPartN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, 
    S1: GemmNode<T, At, Bt, Ct>, S2: GemmNode<T, At, Bt, Ct>> {
    child1: S1,
    child2: S2,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S1, S2>
    GemmNode<T, At, Bt, Ct> for FirstDiffPartN<T, At, Bt, Ct, Bsz, S1, S2> 
    where S1: GemmNode<T, At, Bt, Ct>,
          S2: GemmNode<T, At, Bt, Ct> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr:&ThreadInfo<T>) -> () {
        let n_save = b.push_x_view(Bsz::to_usize());
        c.push_x_view(Bsz::to_usize());
        
        let mut i = 0;
        while i < n_save {
            b.slide_x_view_to(i, Bsz::to_usize());
            c.slide_x_view_to(i, Bsz::to_usize());

            if i == 0 { 
                self.child1.run(a, b, c, thr);
            } else {
                self.child2.run(a, b, c, thr);
            }
            i += Bsz::to_usize();
        }
        
        b.pop_x_view();
        c.pop_x_view();
    }
    fn new() -> Self {
            FirstDiffPartN{ child1: S1::new(), child2: S2::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S1::hierarchy_description();
        child_desc.push(AlgorithmStep::N{ bsz: Bsz::to_usize() });
        child_desc
    }
}

pub struct FirstDiffPartK<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned,
    S1: GemmNode<T, At, Bt, Ct>, S2: GemmNode<T, At, Bt, Ct>> {
    child1: S1,
    child2: S2,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _bszt: PhantomData<Bsz>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S1, S2>
    GemmNode<T, At, Bt, Ct> for FirstDiffPartK<T, At, Bt, Ct, Bsz, S1, S2>
    where S1: GemmNode<T, At, Bt, Ct>,
          S2: GemmNode<T, At, Bt, Ct> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let k_save = a.push_x_view(Bsz::to_usize());
        b.push_y_view(Bsz::to_usize());

        let beta_save = c.get_scalar();

        let mut i = 0;
        while i < k_save  {
            a.slide_x_view_to(i, Bsz::to_usize());
            b.slide_y_view_to(i, Bsz::to_usize());

            if i == 0 { 
                self.child1.run(a, b, c, thr);
            } else {
                self.child2.run(a, b, c, thr);
            }
            
            i += Bsz::to_usize();
            c.set_scalar(T::one());
        }

        a.pop_x_view();
        b.pop_y_view();
        c.set_scalar(beta_save);
    }
    fn new() -> Self {
        FirstDiffPartK{ child1: S1::new(), child2: S2::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut child_desc = S1::hierarchy_description();
        child_desc.push(AlgorithmStep::K{ bsz: Bsz::to_usize() });
        child_desc
    }
}
