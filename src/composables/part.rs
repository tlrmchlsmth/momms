use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::GemmNode;
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
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>> PartM<T,At,Bt,Ct,Bsz,S> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartM<T,At,Bt,Ct,Bsz,S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T> ) -> () {
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
    fn new( ) -> PartM<T,At,Bt,Ct,Bsz,S>{
            PartM{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
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
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>> PartN<T,At,Bt,Ct,Bsz,S> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartN<T,At,Bt,Ct,Bsz,S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr:&ThreadInfo<T> ) -> () {
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
    fn new( ) -> PartN<T, At, Bt, Ct, Bsz, S>{
            PartN{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
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
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>> PartK<T,At,Bt,Ct,Bsz,S> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Bsz: Unsigned, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartK<T,At,Bt,Ct,Bsz,S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T> ) -> () {
        let k_save = a.push_x_view(Bsz::to_usize());
        b.push_y_view(Bsz::to_usize());
        
        let mut i = 0;
        while i < k_save  {
            a.slide_x_view_to(i, Bsz::to_usize());
            b.slide_y_view_to(i, Bsz::to_usize());

            self.child.run(a, b, c, thr);
            i += Bsz::to_usize();
        }

        a.pop_x_view();
        b.pop_y_view();
    }
    fn new( ) -> PartK<T, At, Bt, Ct, Bsz, S>{
        PartK{ child: S::new(), _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _bszt: PhantomData }
    }
}
