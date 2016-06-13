use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
use core::marker::{PhantomData};

pub trait Copier <T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    fn pack( &self, a: &At, a_pack: &mut Apt );
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
    default fn pack( &self, a: &At, a_pack: &mut Apt ) {
        a_pack.copy_from( a );
    }
}

impl<T: Scalar> Copier<T,Matrix<T>,ColumnPanelMatrix<T>> for Packer<T, Matrix<T>, ColumnPanelMatrix<T>> {
    fn pack( &self, a: &Matrix<T>, a_pack: &mut ColumnPanelMatrix<T> ) {
        for panel in 0..a_pack.get_n_panels() {
            for y in 0..a_pack.height() {
                for i in 0..a_pack.get_panel_w() {
                    a_pack.set( y, panel+i, a.get(y, panel+i) );
                }
            }
        }
    }
}

impl<T: Scalar> Copier<T,Matrix<T>,RowPanelMatrix<T>> for Packer<T, Matrix<T>, RowPanelMatrix<T>> {
    fn pack( &self, a: &Matrix<T>, a_pack: &mut RowPanelMatrix<T> ) {
        for panel in 0..a_pack.get_n_panels() {
            for x in 0..a_pack.width() {
                for i in 0..a_pack.get_panel_h() {
                    a_pack.set( panel+i, x, a.get(panel+i, x) );
                }
            }
        }
    }
}
