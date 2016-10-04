use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::GemmNode;
use core::marker::PhantomData;

pub struct PartM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartM<T,At,Bt,Ct,S> {
    pub fn new( bsz: usize, child: S ) -> PartM<T, At, Bt, Ct,S>{
            PartM{ bsz: bsz, child: child, 
                   _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartM<T,At,Bt,Ct,S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T> ) -> () {
        let m_save = a.push_y_view(self.bsz);
        c.push_y_view(self.bsz);
        
        let mut i = 0;
        while i < m_save  {
            a.slide_y_view_to(i, self.bsz);
            c.slide_y_view_to(i, self.bsz);
            
            self.child.run(a, b, c, thr);
            i += self.bsz;
        }

        a.pop_y_view();
        c.pop_y_view();
    }
    #[inline(always)]
    unsafe fn shadow( &self ) -> Self where Self: Sized {
        PartM{ bsz: self.bsz, child: self.child.shadow(), 
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}

pub struct PartN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartN<T,At,Bt,Ct,S> {
    pub fn new( bsz: usize, child: S ) -> PartN<T, At, Bt, Ct, S>{
            PartN{ bsz: bsz, child: child, 
                   _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartN<T,At,Bt,Ct,S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr:&ThreadInfo<T> ) -> () {
        let n_save = b.push_x_view(self.bsz);
        c.push_x_view(self.bsz);
        
        let mut i = 0;
        while i < n_save {
            b.slide_x_view_to(i, self.bsz);
            c.slide_x_view_to(i, self.bsz);

            self.child.run(a, b, c, thr);
            i += self.bsz;
        }
        
        b.pop_x_view();
        c.pop_x_view();
    }
    #[inline(always)]
    unsafe fn shadow( &self ) -> Self where Self: Sized {
        PartN{ bsz: self.bsz, child: self.child.shadow(), 
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}

pub struct PartK<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartK<T,At,Bt,Ct,S> {
    pub fn new( bsz: usize, child: S ) -> PartK<T, At, Bt, Ct, S>{
        PartK{ bsz: bsz, child: child, 
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartK<T,At,Bt,Ct,S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, thr: &ThreadInfo<T> ) -> () {
        let k_save = a.push_x_view(self.bsz);
        b.push_y_view(self.bsz);
        
        let mut i = 0;
        while i < k_save  {
            a.slide_x_view_to(i, self.bsz);
            b.slide_y_view_to(i, self.bsz);

            self.child.run(a, b, c, thr);
            i += self.bsz;
        }

        a.pop_x_view();
        b.pop_y_view();
    }
    #[inline(always)]
    unsafe fn shadow( &self ) -> Self where Self: Sized {
        PartK{ bsz: self.bsz, child: self.child.shadow(), 
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
