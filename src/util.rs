extern crate libc;
extern crate hwloc;

use self::libc::{c_double, int64_t, c_char};
use std::time::Instant;
use std::ffi::{CString};

use thread_comm::ThreadInfo;
use matrix::{Scalar, Mat, Matrix};
use composables::{GemmNode, TripleLoop};

extern{
    fn dgemm_( transa: *const c_char, transb: *const c_char,
               m: *const int64_t, n: *const int64_t, k: *const int64_t,
               alpha: *const c_double, 
               a: *const c_double, lda: *const int64_t,
               b: *const c_double, ldb: *const int64_t,
               beta: *const c_double,
               c: *mut c_double, ldc: *const int64_t );
}

pub fn blas_dgemm( a: &mut Matrix<f64>, b: &mut Matrix<f64>, c: &mut Matrix<f64> ) 
{
    unsafe{ 
        let transa = CString::new("N").unwrap();
        let transb = CString::new("N").unwrap();
        let ap = a.get_mut_buffer();
        let bp = b.get_buffer();
        let cp = c.get_buffer();

        let lda = a.get_column_stride() as int64_t;
        let ldb = b.get_column_stride() as int64_t;
        let ldc = c.get_column_stride() as int64_t;

        let m = c.height() as int64_t;
        let n = b.width() as int64_t;
        let k = a.width() as int64_t;

        let alpha: f64 = 1.0;
        let beta: f64 = 1.0;
    
        dgemm_( transa.as_ptr() as *const c_char, transb.as_ptr() as *const c_char,
                &m, &n, &k,
                &alpha as *const c_double, 
                ap as *const c_double, &lda,
                bp as *const c_double, &ldb,
                &beta as *const c_double,
                cp as *mut c_double, &ldc );
    }
}

pub fn test_c_eq_a_b<T:Scalar, At:Mat<T>, Bt:Mat<T>, Ct:Mat<T>>( a: &mut At, b: &mut Bt, c: &mut Ct ) -> T {
    let mut ref_gemm : TripleLoop = TripleLoop{};

    let m = c.height();
    let n = b.width();
    let k = a.width();

    let mut w : Matrix<T> = Matrix::new(n, 1);
    let mut bw : Matrix<T> = Matrix::new(k, 1);
    let mut abw : Matrix<T> = Matrix::new(m, 1);
    let mut cw : Matrix<T> = Matrix::new(m, 1);
    w.fill_rand();
    cw.fill_zero();
    bw.fill_zero();
    abw.fill_zero();

    //Do bw = Bw, then abw = A*(Bw)   
    unsafe {
        ref_gemm.run( b, &mut w, &mut bw, &ThreadInfo::single_thread() );
        ref_gemm.run( a, &mut bw, &mut abw, &ThreadInfo::single_thread() );
    }

    //Do cw = Cw
    unsafe {
        ref_gemm.run( c, &mut w, &mut cw, &ThreadInfo::single_thread() );
    }
    
    //Cw -= abw
    cw.axpy( T::zero() - T::one(), &abw );
    cw.frosqr()
}

pub fn dur_seconds(start: Instant) -> f64 {
    let dur = start.elapsed();
    let time_secs = dur.as_secs() as f64;
    let time_nanos = dur.subsec_nanos() as f64;
    time_nanos / 1E9 + time_secs
}

pub fn gflops(m: usize, n: usize, k: usize, seconds: f64) -> f64 {
    let nflops = (m * n * k) as f64;
    2.0 * nflops / seconds / 1E9
}

pub fn pin_to_core(core: usize) {
    use self::hwloc::{Topology, CPUBIND_THREAD, ObjectType};
    let mut topo = Topology::new();
    let tid = unsafe { libc::pthread_self() };

    let bind_to = {
        let cores = topo.objects_with_type(&ObjectType::Core).unwrap();
        match cores.get(core) {
            Some(val) => val.cpuset().unwrap(),
            None => panic!("No Core found with id {}", core)
        }
    };
    let _ = topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD);
}
