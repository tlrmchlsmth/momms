use matrix::{Scalar};
use core::marker::{PhantomData};

pub trait GenericXsmmWrapper<T: Scalar> {
    unsafe fn run( m: isize, n: isize, k: isize, 
        alpha: *mut T, 
        a: *mut T, lda: isize,
        b: *mut T, ldb: isize,
        beta: *mut T, 
        c: *mut T, ldc: isize) -> (); 
}

pub struct XsmmWrapper<T: Scalar> {
    _t: PhantomData<T>
}
impl<T: Scalar> GenericXsmmWrapper<T> for XsmmWrapper<T> {
    #[inline(always)]
    default unsafe fn run( _m: isize, _n: isize, _k: isize, _alpha: *mut T, _a: *mut T, _lda: isize,
        _b: *mut T, _ldb: isize, _beta: *mut T, _c: *mut T, _ldc: isize) {
            panic!("Xsmm Wrapper not implemented!");
    }
}

#[cfg(feature="libxsmm")]
pub mod libxsmm 
{
    extern crate libc;
    use self::libc::{ c_double, c_float, int64_t, c_char };
    use kern::xsmm_wrapper::{GenericXsmmWrapper,XsmmWrapper};
    use std::ffi::{CString};

    extern {
        fn libxsmm_dgemm( transa: *const c_char, transb: *const c_char,
                   m: *const int64_t, n: *const int64_t, k: *const int64_t,
                   alpha: *const c_double, 
                   a: *const c_double, lda: *const int64_t,
                   b: *const c_double, ldb: *const int64_t,
                   beta: *const c_double,
                   c: *mut c_double, ldc: *const int64_t );

        fn libxsmm_sgemm( transa: *const c_char, transb: *const c_char,
                   m: *const int64_t, n: *const int64_t, k: *const int64_t,
                   alpha: *const c_float, 
                   a: *const c_float, lda: *const int64_t,
                   b: *const c_float, ldb: *const int64_t,
                   beta: *const c_float,
                   c: *mut c_float, ldc: *const int64_t );
    }

    impl GenericXsmmWrapper<f64> for XsmmWrapper<f64> {
        #[inline(always)]
        unsafe fn run( m: isize, n: isize, k: isize, 
            alpha: *mut f64, 
            a: *mut f64, lda: isize,
            b: *mut f64, ldb: isize,
            beta: *mut f64, 
            c: *mut f64, ldc: isize) -> () 
        {
                let trans = CString::new("N").unwrap();

                libxsmm_dgemm( trans.as_ptr() as *const c_char, trans.as_ptr() as *const c_char,
                    &(m as int64_t), &(n as int64_t), &(k as int64_t),
                    alpha as *const c_double,
                    a as *const c_double, &(lda as int64_t),
                    b as *const c_double, &(ldb as int64_t),
                    beta as *const c_double,
                    c as *mut c_double, &(ldc as int64_t));
        }
    }
    impl GenericXsmmWrapper<f32> for XsmmWrapper<f32> {
        #[inline(always)]
        unsafe fn run( m: isize, n: isize, k: isize, 
            alpha: *mut f32, 
            a: *mut f32, lda: isize,
            b: *mut f32, ldb: isize,
            beta: *mut f32, 
            c: *mut f32, ldc: isize) -> () 
        {
                let trans = CString::new("N").unwrap();

                libxsmm_sgemm( trans.as_ptr() as *const c_char, trans.as_ptr() as *const c_char,
                    &(m as int64_t), &(n as int64_t), &(k as int64_t),
                    alpha as *const c_float,
                    a as *const c_float, &(lda as int64_t),
                    b as *const c_float, &(ldb as int64_t),
                    beta as *const c_float,
                    c as *mut c_float, &(ldc as int64_t));
        }
    }
}
