use core::ptr::{self};
use core::marker::PhantomData;
use core::cmp;

use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix,ResizableBuffer};
use typenum::Unsigned;
use thread_comm::ThreadInfo;
use composables::GemmNode;

//This trait exists so that Packer has a type to specialize over.
//Yes this is stupid.
pub trait Copier <T: Scalar, At: Mat<T>, Apt: Mat<T>> {
        fn pack( &self, a: &At, a_pack: &mut Apt, thr: &ThreadInfo<T> );
}


pub struct Packer<T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _apt: PhantomData<Apt>,
}
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Packer<T, At, Apt> {
    pub fn new() -> Packer<T, At, Apt> {
        Packer{ _t: PhantomData, _at: PhantomData, _apt: PhantomData } 
    }
}
    
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Copier<T, At, Apt> 
    for Packer<T, At, Apt> {
    default fn pack( &self, a: &At, a_pack: &mut Apt, thr: &ThreadInfo<T> ) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        let cols_per_thread = (a.width()-1) / thr.num_threads() + 1;
        let start = cols_per_thread * thr.thread_id();
        let end = cmp::min(a.width(), start+cols_per_thread);

        for x in start..end {
            for y in 0..a.height() { 
                a_pack.set(y, x, a.get(y,x));
            }
        }
    }
}

impl<T: Scalar, PW: Unsigned> Copier<T, Matrix<T>, ColumnPanelMatrix<T, PW>> 
    for Packer<T, Matrix<T>, ColumnPanelMatrix<T, PW>> {
    fn pack( &self, a: &Matrix<T>, a_pack: &mut ColumnPanelMatrix<T, PW>, thr: &ThreadInfo<T> ) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        unsafe {
            let ap = a.get_buffer();
            let cs_a = a.get_column_stride();
            let rs_a = a.get_row_stride();

            let panels_per_thread = (a_pack.get_n_panels()-1) / thr.num_threads() + 1;
            let start = panels_per_thread * thr.thread_id();
            let end = cmp::min(a_pack.get_n_panels(),
                start+panels_per_thread);

            for panel in start..end-1 {
                let p = a_pack.get_panel(panel);
                let ap1 = ap.offset((panel * PW::to_usize() * cs_a) as isize);

                for y in 0..a_pack.height() {
                    for i in 0..PW::to_usize() {
                        let alpha = ptr::read(ap1.offset((y*rs_a + i*cs_a) as isize));
                        ptr::write( p.offset((y*PW::to_usize() + i) as isize), alpha );
                    }
                }
            }
            //Handle the last panel separately, since it might have fewer than panel width number of columns
            let last_panel_w = if end * PW::to_usize() > a_pack.width() {
                a_pack.width() - (end-1)*PW::to_usize()
            } else {
                PW::to_usize()
            };
            let p = a_pack.get_panel(end-1); 
            let ap1 = ap.offset(((end-1) * PW::to_usize() * cs_a) as isize);
            for y in 0..a_pack.height() {
                for i in 0..last_panel_w {
                    let alpha = ptr::read(ap1.offset((y*rs_a + i*cs_a)as isize));
                    ptr::write( p.offset((y*PW::to_usize() + i) as isize), alpha );
                }
            }
        }
    }
}

impl<T: Scalar, PH: Unsigned> Copier<T, Matrix<T>, RowPanelMatrix<T, PH>> 
    for Packer<T, Matrix<T>, RowPanelMatrix<T, PH>> {
    fn pack( &self, a: &Matrix<T>, a_pack: &mut RowPanelMatrix<T, PH>, thr: &ThreadInfo<T> ) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        unsafe {
            let ap = a.get_buffer();
            let cs_a = a.get_column_stride();
            let rs_a = a.get_row_stride();

            let panels_per_thread = (a_pack.get_n_panels()-1) / thr.num_threads() + 1;
            let start = panels_per_thread * thr.thread_id();
            let end = cmp::min(a_pack.get_n_panels(),
                start+panels_per_thread);

            for panel in start..end {
                let p = a_pack.get_panel(panel); 
                let ap1 = ap.offset((panel * PH::to_usize() * rs_a) as isize); 

                for x in 0..a_pack.width() {
                    for i in 0..PH::to_usize() {
                        let alpha = ptr::read(ap1.offset((x*cs_a + i*rs_a) as isize));
                        ptr::write( p.offset((x*PH::to_usize() + i) as isize), alpha );
                    }
                }
            }
            //Handle the last panel separately, since it might have fewer than panel width number of columns
            let last_panel_h = if end * PH::to_usize() > a_pack.height() {
                a_pack.height() - (end-1)*PH::to_usize()
            } else {
                PH::to_usize()
            };
            let p = a_pack.get_panel(end-1); 
            let ap1 = ap.offset(((end-1) * PH::to_usize() * rs_a) as isize); 
            for x in 0..a_pack.width() {
                for i in 0..last_panel_h {
                    let alpha = ptr::read(ap1.offset((x*cs_a + i*rs_a)as isize));
                    ptr::write( p.offset((x*PH::to_usize() + i) as isize), alpha );
                }
            }
        }
    }
}

pub struct PackA<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, APt: Mat<T>, 
    S: GemmNode<T, APt, Bt, Ct>> {
    child: S,
    packer: Packer<T, At, APt>,
    a_pack: APt,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, APt: Mat<T>, S: GemmNode<T, APt, Bt, Ct>> PackA <T,At,Bt,Ct,APt,S> 
    where APt: ResizableBuffer<T> {
    pub fn new( child: S ) -> PackA<T, At, Bt, Ct, APt, S>{
        PackA{ child: child, 
               a_pack: APt::empty(), packer: Packer::new(),
               _bt: PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, APt: Mat<T>, S: GemmNode<T, APt, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackA<T, At, Bt, Ct, APt, S>
    where APt: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T> ) -> () {
        thr.barrier();
        if self.a_pack.capacity() < APt::capacity_for(a) {
            if thr.thread_id() == 0 {
                self.a_pack.aquire_buffer_for(APt::capacity_for(a));
            }
            else {
                self.a_pack.set_capacity( APt::capacity_for(a) );
            }
            self.a_pack.send_alias( thr );
        }
        self.a_pack.resize_to( a );
        self.packer.pack( a, &mut self.a_pack, thr );
        thr.barrier();
        self.child.run(&mut self.a_pack, b, c, thr);
    }
    #[inline(always)]
    unsafe fn shadow( &self ) -> Self where Self: Sized {
        PackA{ child: self.child.shadow(), 
               a_pack: APt::empty(), 
               packer: Packer::new(),
               _bt:PhantomData, _ct: PhantomData }
    }
}

pub struct PackB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, 
    S: GemmNode<T, At, BPt, Ct>> {
    child: S,
    packer: Packer<T, Bt, BPt>,
    b_pack: BPt,
    _at: PhantomData<At>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, S: GemmNode<T, At, BPt, Ct>> PackB <T,At,Bt,Ct,BPt,S> 
    where BPt: ResizableBuffer<T> {
    pub fn new( child: S ) -> PackB<T, At, Bt, Ct, BPt, S>{
        PackB{ child: child, 
               b_pack: BPt::empty(), packer: Packer::new(),
               _at:PhantomData, _ct: PhantomData }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, S: GemmNode<T, At, BPt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackB<T, At, Bt, Ct, BPt, S>
    where BPt: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T> ) -> () {
        thr.barrier();
        if self.b_pack.capacity() < BPt::capacity_for(b) {
            if thr.thread_id() == 0 {
                self.b_pack.aquire_buffer_for(BPt::capacity_for(b));
            }
            else {
                self.b_pack.set_capacity( BPt::capacity_for(b) );
            }
            self.b_pack.send_alias( thr );
        }
        self.b_pack.resize_to( b );
        self.packer.pack( b, &mut self.b_pack, thr );
        thr.barrier();
        self.child.run(a, &mut self.b_pack, c, thr);
    }
    #[inline(always)]
    unsafe fn shadow( &self ) -> Self where Self: Sized {
        PackB{ child: self.child.shadow(), 
               b_pack: BPt::empty(), 
               packer: Packer::new(),
               _at:PhantomData, _ct: PhantomData }
    }
}
