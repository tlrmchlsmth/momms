use matrix::{Scalar,Mat,Matrix,RoCM};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{Unsigned,U4,U6,U8,U12};

extern crate libc;
use self::libc::{ c_double, int64_t };
extern{
    fn bli_dgemm_asm_6x8 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
    fn bli_dgemm_asm_4x12 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
    fn bli_dgemm_asm_12x4 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}

pub struct Ukernel<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
    _mrt: PhantomData<Mr>,
    _nrt: PhantomData<Nr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Mr: Unsigned, Nr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct, Mr, Nr> {
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
    fn new( ) -> Ukernel<T, At, Bt, Ct, Mr, Nr> { 
        Ukernel{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData, _mrt: PhantomData, _nrt: PhantomData } 
    }   
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        Vec::new()
    }   
}

impl<At: Mat<f64>, Bt: Mat<f64>, Ct: Mat<f64>> 
    GemmNode<f64, At, Bt, Ct> for Ukernel<f64, At, Bt, Ct, U8, U6>
    where At: RoCM<f64>, Bt: RoCM<f64>, Ct: RoCM<f64>
{
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<f64> ) -> () {
        debug_assert!(c.height() <= U6::to_usize());
        debug_assert!(c.width() <= U8::to_usize());
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let rs_c = c.get_leaf_rs();
        let cs_c = c.get_leaf_cs();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        if c.height() == U6::to_usize() && c.width() == U8::to_usize() {
            bli_dgemm_asm_6x8 (
                a.width() as int64_t,
                &mut alpha as *mut c_double,
                ap as *mut c_double,
                bp as *mut c_double,
                &mut beta as *mut c_double,
                cp as *mut c_double,
                rs_c as int64_t, cs_c as int64_t );
        }
        else {
            let mut t : Matrix<f64> = Matrix::new( U6::to_usize(), U8::to_usize() );
            let tp = t.get_mut_buffer();
            let rs_t = t.get_row_stride();
            let cs_t = t.get_column_stride();
            beta = 0.0;

            bli_dgemm_asm_6x8 (
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


impl<At: Mat<f64>, Bt: Mat<f64>, Ct: Mat<f64>> 
    GemmNode<f64, At, Bt, Ct> for Ukernel<f64, At, Bt, Ct, U12, U4>
    where At: RoCM<f64>, Bt: RoCM<f64>, Ct: RoCM<f64>
{
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<f64> ) -> () {
        debug_assert!(c.height() <= U4::to_usize());
        debug_assert!(c.width() <= U12::to_usize());
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let rs_c = c.get_leaf_rs();
        let cs_c = c.get_leaf_cs();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        if c.height() == U4::to_usize() && c.width() == U12::to_usize() {
            bli_dgemm_asm_4x12 (
                a.width() as int64_t,
                &mut alpha as *mut c_double,
                ap as *mut c_double,
                bp as *mut c_double,
                &mut beta as *mut c_double,
                cp as *mut c_double,
                rs_c as int64_t, cs_c as int64_t );
        }
        else {
            let mut t : Matrix<f64> = Matrix::new( U4::to_usize(), U12::to_usize() );
            let tp = t.get_mut_buffer();
            let rs_t = t.get_row_stride();
            let cs_t = t.get_column_stride();
            beta = 0.0;

            bli_dgemm_asm_4x12 (
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
