#[allow(dead_code, non_snake_case, non_camel_case_types, non_upper_case_globals)]
#[cfg(feature="blis")]
pub mod blis_types 
{
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use matrix::{Scalar};
use core::marker::{PhantomData};
use typenum::{Unsigned};




pub trait GenericUkernelWrapper<Mr: Unsigned, Nr: Unsigned, T: Scalar> {
    unsafe fn run( k: isize, alpha: *mut T, a: *mut T, b: *mut T, beta: *mut T, c: *mut T, rs_c: isize, cs_c: isize) -> (); 
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
    default unsafe fn run( _: isize, _: *mut T, _: *mut T, _: *mut T, _: *mut T, _: *mut T, _: isize, _: isize) {
        panic!("Ukernel Wrapper not implemented for Mr {} Nr {} and this datatype!", Mr::to_usize(), Nr::to_usize());
    }
}

#[cfg(feature="hsw")]
pub mod hsw
{
    extern crate libc;
    use self::libc::{ c_double, int64_t };
    use typenum::{U4,U6,U8,U12};
    use kern::ukernel_wrapper::{GenericUkernelWrapper,UkernelWrapper};
    use kern::ukernel_wrapper::blis_types::{self,auxinfo_t,inc_t};

    //Haswell ukernels
    extern{
        fn bli_dgemm_haswell_asm_6x8 (k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t,
            auxinfo: *mut auxinfo_t) -> (); 
        fn bli_dgemm_haswell_asm_4x12 (k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t,
            auxinfo: *mut auxinfo_t) -> (); 
        fn bli_dgemm_haswell_asm_12x4 (k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t,
            auxinfo: *mut auxinfo_t) -> (); 
    }

    impl GenericUkernelWrapper<U4, U12, f64> for UkernelWrapper<U4, U12, f64> {
        #[inline(always)]
        unsafe fn run( k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize) {

            let mut info = auxinfo_t{
				schema_a: blis_types::pack_t_BLIS_PACKED_ROW_PANELS,
				schema_b: blis_types::pack_t_BLIS_PACKED_COL_PANELS,
				a_next: a as *mut ::std::os::raw::c_void,
				b_next: b as *mut ::std::os::raw::c_void,
				is_a: 1 as inc_t,
				is_b: 1 as inc_t,
            };
            
            bli_dgemm_haswell_asm_4x12(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t, &mut info as *mut auxinfo_t);
        }
    }
    impl GenericUkernelWrapper<U6, U8, f64> for UkernelWrapper<U6, U8, f64> {
        #[inline(always)]
        unsafe fn run( k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize) {

            let mut info = auxinfo_t{
				schema_a: blis_types::pack_t_BLIS_PACKED_ROW_PANELS,
				schema_b: blis_types::pack_t_BLIS_PACKED_COL_PANELS,
				a_next: a as *mut ::std::os::raw::c_void,
				b_next: b as *mut ::std::os::raw::c_void,
				is_a: 1 as inc_t,
				is_b: 1 as inc_t,
            };
            
            bli_dgemm_haswell_asm_6x8(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t, &mut info as *mut auxinfo_t);
        }
    }
    impl GenericUkernelWrapper<U12, U4, f64> for UkernelWrapper<U12, U4, f64> {
        #[inline(always)]
        unsafe fn run( k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize) {

            let mut info = auxinfo_t{
				schema_a: blis_types::pack_t_BLIS_PACKED_ROW_PANELS,
				schema_b: blis_types::pack_t_BLIS_PACKED_COL_PANELS,
				a_next: a as *mut ::std::os::raw::c_void,
				b_next: b as *mut ::std::os::raw::c_void,
				is_a: 1 as inc_t,
				is_b: 1 as inc_t,
            };

            bli_dgemm_haswell_asm_12x4(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t, &mut info as *mut auxinfo_t);
        }
    }
}

#[cfg(feature="snb")]
pub mod snb
{
    extern crate libc;
    use self::libc::{ c_double, int64_t };
    use typenum::{U4,U8};
    use kern::ukernel_wrapper::{GenericUkernelWrapper,UkernelWrapper};
    use kern::ukernel_wrapper::blis_types::{self,auxinfo_t,inc_t};

    //Haswell ukernels
    extern{
        fn bli_dgemm_sandybridge_int_8x4 (k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t,
            auxinfo: *mut auxinfo_t) -> (); 
    }

    impl GenericUkernelWrapper<U8, U4, f64> for UkernelWrapper<U8, U4, f64> {
        #[inline(always)]
        unsafe fn run( k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize) {

            let mut info = auxinfo_t{
				schema_a: blis_types::pack_t_BLIS_PACKED_ROW_PANELS,
				schema_b: blis_types::pack_t_BLIS_PACKED_COL_PANELS,
				a_next: a as *mut ::std::os::raw::c_void,
				b_next: b as *mut ::std::os::raw::c_void,
				is_a: 1 as inc_t,
				is_b: 1 as inc_t,
            };

            bli_dgemm_sandybridge_int_8x4(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t, &mut info as *mut auxinfo_t);
        }
    }
}

#[cfg(feature="knl")]
pub mod knl
{
	extern crate libc;
    use self::libc::{ c_double, int64_t };
    use typenum::{U24,U8};
    use kern::ukernel_wrapper::{GenericUkernelWrapper,UkernelWrapper};
    use kern::ukernel_wrapper::blis_types::{auxinfo_t,inc_t};

	// KNL ukernels
	extern{
        fn bli_dgemm_knl_asm_24x8 (k: int64_t,
            alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, 
            c: *mut f64, rs_c: int64_t, cs_c: int64_t,
            auxinfo: *mut auxinfo_t) -> (); 
    }

    impl GenericUkernelWrapper<U24, U8, f64> for UkernelWrapper<U24, U8, f64> {
        #[inline(always)]
        unsafe fn run( k: isize, alpha: *mut f64, a: *mut f64, b: *mut f64, beta: *mut f64, c: *mut f64, rs_c: isize, cs_c: isize) {

            let mut info = auxinfo_t{
				schema_a: blis_types::pack_t_BLIS_PACKED_ROW_PANELS,
				schema_b: blis_types::pack_t_BLIS_PACKED_COL_PANELS,
				a_next: a as *mut ::std::os::raw::c_void,
				b_next: b as *mut ::std::os::raw::c_void,
				is_a: 1 as inc_t,
				is_b: 1 as inc_t,
            };

            bli_dgemm__knl_asm_24x8(k as int64_t, alpha as *mut c_double, a as *mut c_double, b as *mut c_double,
                beta as *mut c_double, c as *mut c_double, rs_c as int64_t, cs_c as int64_t, &mut info as *mut auxinfo_t);
        }
    }

}
