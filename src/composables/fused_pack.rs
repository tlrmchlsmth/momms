use core::marker::PhantomData;
use matrix::{Scalar,Mat,PackPair,ResizableBuffer};
//use typenum::Unsigned;
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};

pub struct DelayedPackA<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Apt: Mat<T>, 
    S: GemmNode<T, PackPair<T,At,Apt>, Bt, Ct>> {
    child: S,
    a_pack: Apt,
    algo_desc: Vec<AlgorithmStep>, 
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Apt: Mat<T>, S> DelayedPackA <T,At,Bt,Ct,Apt,S> 
    where Apt: ResizableBuffer<T>,
          S: GemmNode<T, PackPair<T, At, Apt>, Bt, Ct> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Apt: Mat<T>, S>
    GemmNode<T, At, Bt, Ct> for DelayedPackA<T, At, Bt, Ct, Apt, S>
    where Apt: ResizableBuffer<T>,
          S: GemmNode<T, PackPair<T, At, Apt>, Bt, Ct> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        let capacity_for_apt = Apt:: capacity_for(a, y_marker, x_marker, &self.algo_desc);
        thr.barrier();

        //Check if we need to resize packing buffer
        if self.a_pack.capacity() < capacity_for_apt {
            if thr.thread_id() == 0 {
                self.a_pack.aquire_buffer_for(capacity_for_apt);
            }
            else {
                self.a_pack.set_capacity(capacity_for_apt);
            }
            self.a_pack.send_alias(thr);
        }

        //Logically resize the a_pack matrix
        self.a_pack.resize_to(a, y_marker, x_marker, &self.algo_desc);
        let mut pair = PackPair::new(a.make_alias(), self.a_pack.make_alias());
        self.child.run(&mut pair, b, c, thr);
    }
    fn new() -> Self {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        DelayedPackA{ child: S::new(), 
               a_pack: Apt::empty(y_marker, x_marker, &algo_desc), algo_desc: algo_desc,
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}

pub struct DelayedPackB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bpt: Mat<T>, 
    S: GemmNode<T, At, PackPair<T, Bt, Bpt>, Ct>> {
    child: S,
    b_pack: Bpt,
    algo_desc: Vec<AlgorithmStep>, 
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bpt: Mat<T>, S> DelayedPackB <T,At,Bt,Ct,Bpt,S> 
    where Bpt: ResizableBuffer<T>,
          S: GemmNode<T, At, PackPair<T, Bt, Bpt>, Ct> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bpt: Mat<T>, S>
    GemmNode<T, At, Bt, Ct> for DelayedPackB<T, At, Bt, Ct, Bpt, S>
    where Bpt: ResizableBuffer<T>,
          S: GemmNode<T, At, PackPair<T, Bt, Bpt>, Ct> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        let capacity_for_bpt = Bpt:: capacity_for(b, y_marker, x_marker, &self.algo_desc);

        thr.barrier();

        //Check if we need to resize packing buffer
        if self.b_pack.capacity() < capacity_for_bpt {
            if thr.thread_id() == 0 {
                self.b_pack.aquire_buffer_for(capacity_for_bpt);
            }
            else {
                self.b_pack.set_capacity(capacity_for_bpt);
            }
            self.b_pack.send_alias(thr);
        }

        //Logically resize the c_pack matrix
        self.b_pack.resize_to(b, y_marker, x_marker, &self.algo_desc);
        let mut pair = PackPair::new(b.make_alias(), self.b_pack.make_alias());
        
        self.child.run(a, &mut pair, c, thr);
    }
    fn new() -> Self {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};

        DelayedPackB{ child: S::new(), 
               b_pack: Bpt::empty(y_marker, x_marker, &algo_desc), algo_desc: algo_desc,
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}

pub struct UnpairA<T: Scalar, At: Mat<T>, Apt: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, 
    S: GemmNode<T, Apt, Bt, Ct>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _apt: PhantomData<Apt>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S>
    GemmNode<T, PackPair<T, At, Apt>, Bt, Ct> for UnpairA<T, At, Apt, Bt, Ct, S>
    where S: GemmNode<T, Apt, Bt, Ct> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut PackPair<T, At, Apt>, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        self.child.run( &mut a.ap, b, c, thr);
    }
    fn new() -> Self {
        UnpairA{ child: S::new(), 
                 _t: PhantomData, _at: PhantomData, _apt: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}

pub struct UnpairB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Bpt: Mat<T>, Ct: Mat<T>, 
    S: GemmNode<T, At, Bpt, Ct>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _bpt: PhantomData<Bpt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Bpt: Mat<T>, Ct: Mat<T>, S>
    GemmNode<T, At, PackPair<T, Bt, Bpt>, Ct> for UnpairB<T, At, Bt, Bpt, Ct, S>
    where S: GemmNode<T, At, Bpt, Ct> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut PackPair<T, Bt, Bpt>, c: &mut Ct, thr: &ThreadInfo<T>) -> () {
        self.child.run( a, &mut b.ap, c, thr);
    }
    fn new() -> Self {
        UnpairB{ child: S::new(), 
                 _t: PhantomData, _at: PhantomData, _bt: PhantomData, _bpt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}

pub struct UnpairC<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Cpt: Mat<T>,
    S: GemmNode<T, At, Bt, Cpt>> {
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _cpt: PhantomData<Cpt>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Cpt: Mat<T>, S>
    GemmNode<T, At, Bt, PackPair<T, Ct, Cpt>> for UnpairC<T, At, Bt, Ct, Cpt, S>
    where S: GemmNode<T, At, Bt, Cpt> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut PackPair<T, Ct, Cpt>, thr: &ThreadInfo<T>) -> () {
        self.child.run( a, b, &mut c.ap, thr);
    }
    fn new() -> Self {
        UnpairC{ child: S::new(), 
                 _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _cpt: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}
