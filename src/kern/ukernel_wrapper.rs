use matrix::{Scalar};
use core::marker::{PhantomData};
use typenum::{Unsigned};


pub trait GenericUkernelWrapper<Mr: Unsigned, Nr: Unsigned, T: Scalar> {
    fn run(  k: isize, alpha: *mut T, a: *mut T, b: *mut T, beta: *mut T, c: *mut T, rs_c: isize, cs_c: isize ) -> (); 
}

pub struct UkernelWrapper<Mr: Unsigned, Nr: Unsigned, T: Scalar> {
    _t: PhantomData<T>,
    _mr: PhantomData<Mr>,
    _nr: PhantomData<Nr>,
}
impl<Mr: Unsigned, Nr: Unsigned, T: Scalar> UkernelWrapper<Mr, Nr, T> {
}
impl<Mr: Unsigned, Nr: Unsigned, T: Scalar> GenericUkernelWrapper<Mr, Nr, T> for UkernelWrapper<Mr, Nr, T> {
    #[inline(always)]
    default fn run(  _: isize, _: *mut T, _: *mut T, _: *mut T, _: *mut T, _: *mut T, _: isize, _: isize ) {
        panic!("Ukernel Wrapper not implemented for Mr {} Nr {} and this datatype!", Mr::to_usize(), Nr::to_usize());
    }
}

#[cfg(hsw)]
pub use hsw;
#[cfg(hsw)]
mod hsw
{
    extern crate libc;
    use self::libc::{ c_double, int64_t };
    use typenum::{U1,U4,U6,U8,U12};

    //Haswell ukernels
    extern{
        fn bli_dgemm_asm_6x8 ( k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t ) -> (); 
        fn bli_dgemm_asm_4x12 ( k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t ) -> (); 
        fn bli_dgemm_asm_12x4 ( k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t ) -> (); 
    }

    impl GenericUkernelWrapper<U4, U12, f64> for UkernelWrapper<U4, U12, f64> {
        #[inline(always)]
        fn run(  k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize ) {
            bli_dgemm_asm_4x12(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                              beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t);
        }
    }
    impl GenericUkernelWrapper<U6, U8, f64> for UkernelWrapper<U6, U8, f64> {
        #[inline(always)]
        fn run(  k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize ) {
            bli_dgemm_asm_6x8(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                              beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t);
        }
    }
    impl GenericUkernelWrapper<U12, U4, f64> for UkernelWrapper<U12, U4, f64> {
        #[inline(always)]
        fn run(  k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize ) {
            bli_dgemm_asm_12x4(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                              beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t);
        }
    }
}

#[cfg(snb)]
pub use snb;
#[cfg(snb)]
mod snb
{
    extern crate libc;
    use self::libc::{ c_double, int64_t };
    use typenum::{U1,U4,U8};

    //Haswell ukernels
    extern{
        fn bli_dgemm_int_8x4 ( k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t ) -> (); 
    }

    impl GenericUkernelWrapper<U8, U4, f64> for UkernelWrapper<U8, U4, f64> {
        #[inline(always)]
        fn run(  k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize ) {
            bli_dgemm_int_8x4(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                              beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t);
        }
    }
}
