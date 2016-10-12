use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix,Hierarch};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{Unsigned,U1,U4,U8};

extern crate libc;
use self::libc::{ c_double, int64_t };

extern{
    fn bli_dgemm_asm_8x4 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
    fn bli_dgemm_int_8x4 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}

pub struct Ukernel<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct> {
    #[inline(always)]
    default unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        for z in 0..a.width() {
            for y in 0..c.height() {
                for x in 0..c.width() {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
    }
    fn new( ) -> Ukernel<T, At, Bt, Ct> { 
        Ukernel{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData } 
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        Vec::new()
    }  
}

impl GemmNode<f64, RowPanelMatrix<f64, U8>, ColumnPanelMatrix<f64, U4>, Matrix<f64>> 
    for Ukernel<f64, RowPanelMatrix<f64, U8>, ColumnPanelMatrix<f64, U4>, Matrix<f64>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut RowPanelMatrix<f64, U8>, 
                   b: &mut ColumnPanelMatrix<f64, U4>, 
                   c: &mut Matrix<f64>, 
                   _thr: &ThreadInfo<f64> ) -> () {
        debug_assert!(c.height() <= U8::to_usize());
        debug_assert!(c.width() <= U4::to_usize());


        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();
        let rs_c = c.get_row_stride();
        let cs_c = c.get_column_stride();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        if c.height() == U8::to_usize() && c.width() == U4::to_usize() {
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
        else {
            //TODO: cache c_tmp somewhere!
            let mut t : Matrix<f64> = Matrix::new( U8::to_usize(), U4::to_usize() );
            let tp = t.get_mut_buffer();
            let rs_t = t.get_row_stride();
            let cs_t = t.get_column_stride();
            beta = 0.0;

            bli_dgemm_int_8x4 (
                a.width() as int64_t,
                &mut alpha as *mut c_double,
                ap as *mut c_double,
                bp as *mut c_double,
                &mut beta as *mut c_double,
                tp as *mut c_double,
                rs_t as int64_t, cs_t as int64_t );
    

            t.push_y_view(c.height());
            t.push_x_view(c.width());
            c.axpby_small( 1.0, &t, 1.0 );
        }
    }
}

impl<K: Unsigned>
    GemmNode<f64, Hierarch<f64, U8, K, U1, U8>,
                  Hierarch<f64, K, U4, U4, U1>,
                  Hierarch<f64, U8, U4, U1, U8>> for
    Ukernel<f64, Hierarch<f64, U8, K, U1, U8>,
                 Hierarch<f64, K, U4, U4, U1>,
                 Hierarch<f64, U8, U4, U1, U8>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut Hierarch<f64, U8, K, U1, U8>,
                   b: &mut Hierarch<f64, K, U4, U4, U1>,
                   c: &mut Hierarch<f64, U8, U4, U1, U8>,
                   _thr: &ThreadInfo<f64> ) -> () {
//        assert!(c.height() == self.mr);
//        assert!(c.width() == self.nr);


        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();
//        let rs_c = c.get_row_stride();
//        let cs_c = c.get_column_stride();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        //bli_dgemm_asm_8x4 ( 
        bli_dgemm_int_8x4 (
            a.width() as int64_t,
            &mut alpha as *mut c_double,
            ap as *mut c_double,
            bp as *mut c_double,
            &mut beta as *mut c_double,
            cp as *mut c_double,
            1 as int64_t, 8 as int64_t );
    }
}


impl 
    GemmNode<f64, RowPanelMatrix<f64, U8>, 
                  ColumnPanelMatrix<f64, U4>, 
                  Hierarch<f64, U8, U4, U1, U8>> for
    Ukernel<f64, RowPanelMatrix<f64, U8>, 
                 ColumnPanelMatrix<f64, U4>, 
                 Hierarch<f64, U8, U4, U1, U8>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut RowPanelMatrix<f64, U8>, 
                   b: &mut ColumnPanelMatrix<f64, U4>, 
                   c: &mut Hierarch<f64, U8, U4, U1, U8>,
                   _thr: &ThreadInfo<f64> ) -> () {
//        assert!(c.height() == self.mr);
//        assert!(c.width() == self.nr);
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();
//        let rs_c = c.get_row_stride();
//        let cs_c = c.get_column_stride();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        //bli_dgemm_asm_8x4 ( 
        bli_dgemm_int_8x4 (
            a.width() as int64_t,
            &mut alpha as *mut c_double,
            ap as *mut c_double,
            bp as *mut c_double,
            &mut beta as *mut c_double,
            cp as *mut c_double,
            1 as int64_t, 8 as int64_t );
    }
}
