use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
use core::marker::{PhantomData};
use core::ptr::{self};
use core::cmp;
use typenum::{Unsigned};
use thread::{ThreadInfo};

pub trait Copier <T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    fn pack( &self, a: &At, a_pack: &mut Apt, thr: &ThreadInfo<T> );
}

pub struct Packer<T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _apt: PhantomData<Apt>,
}
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Packer<T, At, Apt> {
    pub fn new() -> Packer<T, At, Apt> {Packer{ _t: PhantomData, _at: PhantomData, _apt: PhantomData } }
}

impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Copier<T,At,Apt> for Packer<T, At, Apt> {
    default fn pack( &self, a: &At, a_pack: &mut Apt, thr: &ThreadInfo<T> ) {
        if thr.thread_id() == 0 {
            a_pack.copy_from( a );
        }
    }
}

impl<T: Scalar, PW: Unsigned> Copier<T, Matrix<T>, ColumnPanelMatrix<T, PW>> 
    for Packer<T, Matrix<T>, ColumnPanelMatrix<T, PW>> {
    fn pack( &self, a: &Matrix<T>, a_pack: &mut ColumnPanelMatrix<T, PW>, thr: &ThreadInfo<T> ) {
        unsafe {
            let ap = a.get_buffer();
            let cs_a = a.get_column_stride();
            let rs_a = a.get_row_stride();

            let panels_per_thread = (a_pack.get_n_panels()-1) / thr.num_threads() + 1;
            let start = panels_per_thread * thr.thread_id();
            let end = cmp::min(a_pack.get_n_panels(),
                start+panels_per_thread);
            
            //TODO: Handle edge case where less than a full panel is there at the end.
            for panel in start..end {
                let p = a_pack.get_panel(panel);
                let h = a_pack.height();
                let ap1 = ap.offset((panel * PW::to_usize() * cs_a) as isize);

                for y in 0..h {
                    for i in 0..PW::to_usize() {
                        let alpha = ptr::read(ap1.offset((y*rs_a + i*cs_a)as isize));
                        ptr::write( p.offset((y*PW::to_usize() + i) as isize), alpha );
                    }
                }
            }
        }
    }
}

impl<T: Scalar, PH: Unsigned> Copier<T, Matrix<T>, RowPanelMatrix<T, PH>> 
    for Packer<T, Matrix<T>, RowPanelMatrix<T, PH>> {
    fn pack( &self, a: &Matrix<T>, a_pack: &mut RowPanelMatrix<T, PH>, thr: &ThreadInfo<T> ) {
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
                let w = a_pack.width();
                let ap1 = ap.offset((panel * PH::to_usize() * rs_a) as isize); 

                for x in 0..w {
                    for i in 0..PH::to_usize() {
                        let alpha = ptr::read(ap1.offset((x*cs_a + i*rs_a) as isize));
                        ptr::write( p.offset((x*PH::to_usize() + i) as isize), alpha );
                    }
                }
            }
        }
    }
}
