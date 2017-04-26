extern crate libc;

use typenum::{U1,U4,U6,U8,U12,Unsigned};
use self::libc::{c_double, int64_t};

extern{
    fn bli_dgemm_asm_6x8 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
    fn bli_dgemm_asm_4x12 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}
