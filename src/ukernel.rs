use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
use core::marker::{PhantomData};
use cntl::{GemmNode};

extern crate libc;
use self::libc::{ c_double, int64_t };

#[link(name = "blis", kind = "static")]
extern{
    fn bli_dgemm_asm_8x4 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
    fn bli_dgemm_int_8x4 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}

pub struct Ukernel<T>{
    mr: usize,
    nr: usize,
    _t: PhantomData<T>,
    
}
impl<T: Scalar> Ukernel<T> {
    pub fn new( mr: usize, nr: usize ) -> Ukernel<T> { Ukernel{ mr: mr, nr: nr, _t: PhantomData::<T> } }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T> {
    #[inline(always)]
    default fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        for z in 0..a.width() {
            for y in 0..self.mr {
                for x in 0..self.nr {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
    }
}
impl<T: Scalar, Ct: Mat<T>> 
    GemmNode<T, RowPanelMatrix<T>, ColumnPanelMatrix<T>, Ct> for Ukernel<T> {
    #[inline(always)]
    default fn run( &mut self, a: &mut RowPanelMatrix<T>, b: &mut ColumnPanelMatrix<T>, c: &mut Ct ) -> () {
        unsafe{
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        for z in 0..a.width() {
            for y in 0..self.mr {
                for x in 0..self.nr {
                    let t = *ap.offset((z*self.mr + y) as isize) * *bp.offset((z*self.nr + x) as isize) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
        }
    }
}

//Todo:
//finish this function, call some inline assembly ukernel.
//
//There needs to be Mat.get_raw_pointer() functions,
//and accessors for the strides

impl GemmNode<f64, RowPanelMatrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>> for Ukernel<f64> {
    #[inline(always)]
    fn run( &mut self, a: &mut RowPanelMatrix<f64>, b: &mut ColumnPanelMatrix<f64>, c: &mut Matrix<f64> ) -> () {
        unsafe{
            let ap = a.get_mut_buffer();
            let bp = b.get_mut_buffer();
            let cp = c.get_mut_buffer();
            let rs_c: usize = c.get_row_stride();
            let cs_c: usize = c.get_column_stride();
            let mut alpha: f64 = 1.0;
            let mut beta: f64 = 1.0;

            //just call blis ukernel
            //bli_dgemm_asm_8x4 ( 
            bli_dgemm_int_8x4 (
                a.width() as int64_t,
                &mut alpha as *mut c_double,
                ap as *mut c_double,
                bp as *mut c_double,
                &mut beta as *mut c_double,
                cp as *mut c_double,
                rs_c as int64_t, cs_c as int64_t );
        }
    }
}

