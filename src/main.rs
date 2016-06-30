#![feature(specialization)]
#![feature(zero_one)]
#![feature(asm)]
#![feature(alloc, heap_api)]

use std::time::{Instant};
extern crate core;
extern crate typenum;

mod matrix;
mod gemm;
mod pack;
mod ukernel;
mod thread;
pub use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
//pub use gemm::{GemmNode,PartM,PartN,PartK,PackArp,PackAcp,PackBrp,PackBcp,TripleLoopKernel};
pub use gemm::{GemmNode,PartM,PartN,PartK,PackA,PackB,TripleLoopKernel};
pub use ukernel::{Ukernel};
pub use thread::{ThreadInfo,SpawnThreads,ParallelM,ThreadsTarget};

use typenum::{U4, U8};

extern crate libc;
use self::libc::{ c_double, int32_t, c_char };
use std::ffi::{ CString };

#[link(name = "blis", kind = "static")]
extern{
    fn dgemm_( transa: *const c_char, transb: *const c_char,
               m: *const int32_t, n: *const int32_t, k: *const int32_t,
               alpha: *const c_double, 
               a: *const c_double, lda: *const int32_t,
               b: *const c_double, ldb: *const int32_t,
               beta: *const c_double,
               c: *mut c_double, ldc: *const int32_t );
}

fn blas_dgemm( a: &mut Matrix<f64>, b: &mut Matrix<f64>, c: &mut Matrix<f64> ) 
{
    unsafe{ 
        let transa = CString::new("N").unwrap();
        let transb = CString::new("N").unwrap();
        let ap = a.get_mut_buffer();
        let bp = b.get_buffer();
        let cp = c.get_buffer();

        let lda = a.get_column_stride() as int32_t;
        let ldb = b.get_column_stride() as int32_t;
        let ldc = c.get_column_stride() as int32_t;

        let m = c.height() as int32_t;
        let n = b.width() as int32_t;
        let k = a.width() as int32_t;

        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
    
        dgemm_( transa.as_ptr() as *const c_char, transb.as_ptr() as *const c_char,
                &m, &n, &k,
                &alpha as *const c_double, 
                ap as *const c_double, &lda,
                bp as *const c_double, &ldb,
                &beta as *const c_double,
                cp as *mut c_double, &ldc );
    }
}

fn test_c_eq_a_b<T:Scalar, At:Mat<T>, Bt:Mat<T>, Ct:Mat<T>>( a: &mut At, b: &mut Bt, c: &mut Ct ) -> T {
    let mut ref_gemm : TripleLoopKernel = TripleLoopKernel{};

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

fn time_sweep_goto() -> ()
{
    let ukernel = Ukernel::new( 8, 4 );
//    let ukernel = TripleLoopKernel::new();
    let loop1: PartM<f64, RowPanelMatrix<f64,U8>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, _> 
        = PartM::new( 8, ukernel);
    let loop2: PartN<f64, RowPanelMatrix<f64,U8>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, _> 
        = PartN::new( 4, loop1 );
    let packa: PackA<f64, Matrix<f64>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, RowPanelMatrix<f64,U8>, _>
        = PackA::new( loop2 );
    let loop3: PartM<f64, Matrix<f64>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, _>
        = PartM::new( 96, packa );
    let packb: PackB<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, ColumnPanelMatrix<f64,U4>, _>
        = PackB::new( loop3 );
    let loop4: PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartK::new( 256, packb );
    let loop5: PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartN::new( 4096, loop4 );

//    let ukernel = TripleLoopKernel::new();
   /* let loop3: PartM<f64, Matrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _>
        = PartM::new( 96, ukernel );
    let packb: PackBcp<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PackBcp::new( 4, loop3 );
    let loop4: PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartK::new( 256, packb );
    let loop5: PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartN::new( 4096, loop4 );*/

//    let packb: PackBcp<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
//        = PackBcp::new( 4, ukernel );
    let par_m : ParallelM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _> 
        = ParallelM::new(ThreadsTarget::TheRest,4, loop5);
    let mut algo : SpawnThreads<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _> 
        = SpawnThreads::new(2,par_m);

    for index in 0..64 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;

        let size = (index + 1) * 16;
        let m = size;
        let n = size;
        let k = size;


        for _nrep in 0..10 {

            let mut a : Matrix<f64> = Matrix::new(m, k);
            let mut b : Matrix<f64> = Matrix::new(k, n);
            let mut c : Matrix<f64> = Matrix::new(m, n);
            a.fill_rand();
            b.fill_rand();
            c.fill_zero();
            
            let start = Instant::now();
            unsafe{
                algo.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            let mut dur = start.elapsed();
            let time_secs = dur.as_secs() as f64;
            let time_nanos = dur.subsec_nanos() as f64;
            let time = time_nanos / 1E9 + time_secs;
            best_time = best_time.min(time);

            let err = test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err = worst_err.max(err);

            let start_blis = Instant::now();
            blas_dgemm( &mut a, &mut b, &mut c);
            dur = start_blis.elapsed();
            let time_secs_blis = dur.as_secs() as f64;
            let time_nanos_blis = dur.subsec_nanos() as f64;
            let time_blis = time_nanos_blis / 1E9 + time_secs_blis;
            best_time_blis = best_time_blis.min(time_blis);

        }
        let nflops = (m * n * k) as f64;
        println!("{}\t{}\t{}\t{}", size, 
                 2.0 * nflops / best_time / 1E9, 
                 2.0 * nflops / best_time_blis / 1E9,
                 format!("{:e}", worst_err.sqrt()));
    }
}

fn main() {
    time_sweep_goto( );
}
