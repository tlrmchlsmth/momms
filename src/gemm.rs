use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
use core::marker::{PhantomData};
use pack::{Copier,Packer};

use std::time::{Duration,Instant};

extern crate core;

pub trait GemmNode<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> ();
}


pub struct PackAcp<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, ColumnPanelMatrix<T>, Bt, Ct>> {
    child: S,
    panel_width: usize,
    packer: Packer<T, At, ColumnPanelMatrix<T>>,
    a_pack: ColumnPanelMatrix<T>,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, ColumnPanelMatrix<T>, Bt, Ct>> 
    PackAcp <T,At,Bt,Ct,S> {
    #[inline(always)]
    pub fn new( panel_width: usize, child: S ) -> PackAcp<T, At, Bt, Ct, S>{
        let matrix = ColumnPanelMatrix::new( 0, 0, panel_width );
        let packer = Packer::new();
        PackAcp{ panel_width: panel_width, child: child, a_pack: matrix, packer: packer,
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, ColumnPanelMatrix<T>, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackAcp<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c:&mut Ct ) -> () {
        self.a_pack.resize( a.height(), a.width() );
        self.packer.pack( a, &mut self.a_pack );
        self.child.run(&mut self.a_pack, b, c);
    }
}

pub struct PackBcp<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, ColumnPanelMatrix<T>, Ct>> {
    child: S,
    panel_width: usize,
    packer: Packer<T, Bt, ColumnPanelMatrix<T>>,
    b_pack: ColumnPanelMatrix<T>,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, ColumnPanelMatrix<T>, Ct>> 
    PackBcp <T,At,Bt,Ct,S> {
    #[inline(always)]
    pub fn new( panel_width: usize, child: S ) -> PackBcp<T, At, Bt, Ct, S>{
        let matrix = ColumnPanelMatrix::new( 0, 0, panel_width );
        let packer = Packer::new();
        PackBcp{ panel_width: panel_width, child: child, b_pack: matrix, packer: packer,
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, ColumnPanelMatrix<T>, Ct>>
    GemmNode<T, At, Bt, Ct> for PackBcp<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        self.b_pack.resize( b.height(), b.width() );
        self.packer.pack( b, &mut self.b_pack );
        self.child.run(a, &mut self.b_pack, c);
    }
}
pub struct PackArp<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, RowPanelMatrix<T>, Bt, Ct>> {
    child: S,
    panel_height: usize,
    packer: Packer<T, At, RowPanelMatrix<T>>,
    a_pack: RowPanelMatrix<T>,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, RowPanelMatrix<T>, Bt, Ct>> 
    PackArp <T,At,Bt,Ct,S> {
    #[inline(always)]
    pub fn new( panel_height: usize, child: S ) -> PackArp<T, At, Bt, Ct, S>{
        let matrix = RowPanelMatrix::new( 0, 0, panel_height );
        let packer = Packer::new();
        PackArp{ panel_height: panel_height, child: child, a_pack: matrix, packer: packer,
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, RowPanelMatrix<T>, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackArp<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        self.a_pack.resize( a.height(), a.width() );
        self.packer.pack( a, &mut self.a_pack );
        self.child.run(&mut self.a_pack, b, c);
    }
}

pub struct PackBrp<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, RowPanelMatrix<T>, Ct>> {
    child: S,
    panel_height: usize,
    packer: Packer<T, Bt, RowPanelMatrix<T>>,
    b_pack: RowPanelMatrix<T>,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, RowPanelMatrix<T>, Ct>> 
    PackBrp <T,At,Bt,Ct,S> {
    pub fn new( panel_height: usize, child: S ) -> PackBrp<T, At, Bt, Ct, S>{
        let matrix = RowPanelMatrix::new( 0, 0, panel_height );
        let packer = Packer::new();
        PackBrp{ panel_height: panel_height, child: child, b_pack: matrix, packer: packer,
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, RowPanelMatrix<T>, Ct>>
    GemmNode<T, At, Bt, Ct> for PackBrp<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        self.b_pack.resize( b.height(), b.width() );
        self.packer.pack( b, &mut self.b_pack );
        self.child.run(a, &mut self.b_pack, c);
    }
}

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
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        let m_save = c.height();
        let ay_off_save = a.off_y();
        let cy_off_save = c.off_y();
        
        let mut i = 0;
        while i < m_save  {
            let bsz_step = core::cmp::min( self.bsz, m_save - i );
            a.set_height( bsz_step );
            a.set_off_y( ay_off_save + i );
            c.set_height( bsz_step );
            c.set_off_y( cy_off_save + i );

            self.child.run(a, b, c);
            i += bsz_step;
        }
        a.set_height( m_save );
        a.set_off_y( ay_off_save );
        c.set_height( m_save );
        c.set_off_y( cy_off_save );
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
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        let n_save = c.width();
        let bx_off_save = b.off_x();
        let cx_off_save = c.off_x();
        
        let mut i = 0;
        while i < n_save  {
            let bsz_step = core::cmp::min( self.bsz, n_save - i );
            b.set_width( bsz_step );
            b.set_off_x( bx_off_save + i );
            c.set_width( bsz_step );
            c.set_off_x( cx_off_save + i );

            self.child.run(a, b, c);
            i += bsz_step;
        }
        b.set_width( n_save );
        b.set_off_x( bx_off_save );
        c.set_width( n_save );
        c.set_off_x( cx_off_save );
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
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        let k_save = a.width();
        let ax_off_save = a.off_x();
        let by_off_save = b.off_y();
        
        let mut i = 0;
        while i < k_save  {
            let bsz_step = core::cmp::min( self.bsz, k_save - i );
            a.set_width( bsz_step );
            a.set_off_x( ax_off_save + i );
            b.set_height( bsz_step );
            b.set_off_y( by_off_save + i );

            self.child.run(a, b, c);
            i += bsz_step;
        }
        a.set_width( k_save );
        a.set_off_x( ax_off_save );
        b.set_height( k_save );
        b.set_off_y( by_off_save );
    }
}

pub struct TripleLoopKernel{}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for TripleLoopKernel {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        //For now, let's do an axpy based gemm
        for x in 0..c.width() {
            for z in 0..a.width() {
                for y in 0..c.height() {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
    }
}
impl TripleLoopKernel {
    pub fn new() -> TripleLoopKernel {
        TripleLoopKernel{}
    }
}
