#[allow(dead_code, non_snake_case, non_camel_case_types, non_upper_case_globals)]
#[cfg(feature="blis")]
pub mod blis_types 
{
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use matrix::{Scalar};
use core::marker::{PhantomData};

pub trait GenericKnmKernelWrapper<T: Scalar, const MR: usize, const NR: usize> {
    unsafe fn run( k: isize, alpha: *mut T, a: *mut T, b: *mut T, beta: *mut T, c: *mut T, rs_c: isize, cs_c: isize) -> (); 
}

pub struct KnmKernelWrapper<T: Scalar, const MR: usize, const NR: usize> {
    _t: PhantomData<T>,
}
impl<T:Scalar, const MR: usize, const NR: usize> KnmKernelWrapper<T, {MR}, {NR}> {
}
impl<T:Scalar, const MR: usize, const NR: usize> GenericKnmKernelWrapper<T, {MR}, {NR}> for KnmKernelWrapper<T, {MR}, {NR}> {
    #[inline(always)]
    default unsafe fn run( _: isize, _: *mut T, _: *mut T, _: *mut T, _: *mut T, _: *mut T, _: isize, _: isize) {
        panic!("KnmKernel Wrapper not implemented for MR {} NR {} and this datatype!", MR, NR);
    }
}

#[cfg(feature="knm")]
pub mod knm
{
    extern crate libc;
    use self::libc::{ c_float, int64_t };
    use kern::knm_kernel_wrapper::{GenericKnmKernelWrapper,KnmKernelWrapper};
   // use kern::knm_kernel_wrapper::blis_types::{self,auxinfo_t,inc_t};

    //Haswell ukernels
    extern{
        fn sgemm_knm_int_16x24 (k: int64_t,
            alpha: *mut f32, a: *mut f32, b: *mut f32, beta: *mut f32, 
            c: *mut f32, rs_c: int64_t, cs_c: int64_t) -> (); 
        fn sgemm_knm_asm_16x24 (k: int64_t,
            alpha: *mut f32, a: *mut f32, b: *mut f32, beta: *mut f32, 
            c: *mut f32, rs_c: int64_t, cs_c: int64_t) -> (); 
    }

    impl GenericKnmKernelWrapper<f32, 16, 24> for KnmKernelWrapper<f32, 16, 24> {
        #[inline(always)]
        unsafe fn run( k: isize, alpha: *mut f32, a: *mut f32, b: *mut f32, beta: *mut f32, c: *mut f32, rs_c: isize, cs_c: isize) {

/*            let mut info = auxinfo_t{
				schema_a: blis_types::pack_t_BLIS_PACKED_ROW_PANELS,
				schema_b: blis_types::pack_t_BLIS_PACKED_COL_PANELS,
				a_next: a as *mut ::std::os::raw::c_void,
				b_next: b as *mut ::std::os::raw::c_void,
				is_a: 1 as inc_t,
				is_b: 1 as inc_t,
            };*/

            sgemm_knm_asm_16x24(k as int64_t, alpha as *mut c_float, a as *mut c_float, b as *mut c_float,
                beta as *mut c_float, c as *mut c_float, rs_c as int64_t, cs_c as int64_t); //, &mut info as *mut auxinfo_t);
        }
    }
}

