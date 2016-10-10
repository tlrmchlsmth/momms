#![feature(specialization)]
#![feature(asm)]
#![feature(alloc, heap_api)]
#![feature(conservative_impl_trait)]

use std::time::{Instant};
extern crate core;
extern crate typenum;

mod matrix;
mod composables;
mod thread_comm;

mod ukernel;
mod triple_loop;

pub use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
pub use composables::{GemmNode,PartM,PartN,PartK,PackA,PackB,SpawnThreads,ParallelM,ParallelN,Nwayer};
pub use thread_comm::ThreadInfo;
pub use ukernel::Ukernel;
pub use triple_loop::TripleLoop;

use typenum::{U4, U8, U4096};

extern crate libc;
use self::libc::{ c_double, int32_t, c_char };
use std::ffi::{ CString };

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

fn time_sweep_goto() -> ()
{
/*    let ukernel = Ukernel::new( 8, 4 );
//    let ukernel = TripleLoop::new();
    let loop1: PartM<f64, RowPanelMatrix<f64,U8>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, U8, _> 
        = PartM::new( ukernel);
    let loop2: PartN<f64, RowPanelMatrix<f64,U8>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, U4, _> 
        = PartN::new( loop1 );

//    let par_n : ParallelN<f64, RowPanelMatrix<f64,U8>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, _> 
//        = ParallelN::new(ThreadsTarget::TheRest, 1, loop2);

    let packa: PackA<f64, Matrix<f64>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, RowPanelMatrix<f64,U8>, _>
        = PackA::new( loop2 );
    let loop3: PartM<f64, Matrix<f64>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, U96, _>
        = PartM::new( packa );

//    let par_m : ParallelM<f64, Matrix<f64>, ColumnPanelMatrix<f64,U4>, Matrix<f64>, _> 
//        = ParallelM::new(ThreadsTarget::TheRest, 8, loop3);

    let packb: PackB<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, ColumnPanelMatrix<f64,U4>, _>
        = PackB::new( loop3 );
    let loop4: PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, U256, _>
        = PartK::new( packb );
    let mut algo: PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, U4096, _>
        = PartN::new( loop4 );

//    let mut algo : SpawnThreads<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _> 
//        = SpawnThreads::new(2,loop5);
//
*/
    let mut algo : TripleLoop = TripleLoop{};

    for index in 0..64 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;

//        let size = (index + 1) * 16;
        let size = (index + 1) * 4;
        let m = size;
        let n = size;
        let k = size;

        let n_reps = 10;
        for _ in 0..n_reps {
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


use typenum::{U16, U32, U48, U64, U96, U128, U256, U1};
pub use matrix::{Hierarch,HBX,HBY,HBLeaf};

//M 8 N 4
/*type HBA = HBX<U64, HBY<U8, HBLeaf>>;
type HBB = HBX<U64, HBY<U64, HBLeaf>>;
type HBC = HBX<U64, HBY<U8, HBLeaf>>;
type HierA<T> = Hierarch<T, HBA, U8, U64, U1, U8>;
type HierB<T> = Hierarch<T, HBA, U64, U64, U1, U64>;
type HierC<T> = Hierarch<T, HBC, U8, U64, U1, U8>;*/
//type HBA = HBX<U64, HBY<U64, HBLeaf>>;
type HBA = HBX<U256, HBY<U96, HBY<U8, HBLeaf>>>;
type HBB = HBY<U256, HBX<U4, HBLeaf>>;
type HBC = HBY<U96, HBX<U4, HBY<U8, HBLeaf>>>;
type HierA<T> = Hierarch<T, HBA, U8, U256, U1, U8>;
type HierB<T> = Hierarch<T, HBB, U256, U4, U4, U1>;
type HierC<T> = Hierarch<T, HBC, U8, U4, U1, U8>;

//sequence:
// k 256, m 96, n 32, k 32, m 8 n 4
type HBA2 = HBX<U256, HBY<U96, HBX<U64, HBY<U8, HBLeaf>>>>;
type HBB2 = HBY<U256, HBX<U96, HBY<U64, HBX<U4, HBLeaf>>>>;
type HBC2 = HBY<U96,  HBX<U96, HBY<U8,  HBX<U4, HBLeaf>>>>;
type HierA2<T> = Hierarch<T, HBA2, U8, U64, U1, U8>;
type HierB2<T> = Hierarch<T, HBB2, U64, U4, U4, U1>;
type HierC2<T> = Hierarch<T, HBC2, U8, U4, U1, U8>;

fn test_hierarch() {

    //let kernel : TripleLoop = TripleLoop{};
/*    let kernel: Ukernel<f64, HierA<f64>, HierB<f64>, HierC<f64>> = Ukernel::new( 8, 4 );
    let loop1: PartM<f64, HierA<f64>, HierB<f64>, HierC<f64>, _> 
        = PartM::new( 8, kernel);
    let loop2: PartN<f64, HierA<f64>, HierB<f64>, HierC<f64>, _> 
        = PartN::new( 4, loop1 );
    let loop3: PartM<f64, HierA<f64>, HierB<f64>, HierC<f64>, _> 
        = PartM::new( 96, loop2 );
    let loop4: PartK<f64, HierA<f64>, HierB<f64>, HierC<f64>, _>
        = PartK::new( 256, loop3 );
    let mut gemm = loop4;*/
    let mut ref_gemm : TripleLoop = TripleLoop{};

/*
    let ukernel2: Ukernel<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>> = Ukernel::new( 8, 4 );
    let loop_r_n: PartN<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U4, _>  
        = PartN::new(ukernel2);
    let loop_r_m: PartM<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U8, _>  
        = PartM::new(loop_r_n);

    let loop_l1_k: PartK<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U64, _>  
        = PartK::new(loop_r_m);
    let loop_l1_n: PartN<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U96, _>  
        = PartN::new(loop_l1_k);

    let loop_l2_m: PartM<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U96, _>  
        = PartM::new(loop_l1_n);
    let loop_l2_k: PartK<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U256, _>  
        = PartK::new(loop_l2_m);
    let mut gemm2 = loop_l2_k;*/
    type Blah = PartK<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U256,
                PartM<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U96,
                PartN<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U96,
                PartK<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U64,
                PartM<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U8,
                PartN<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U4,
                Ukernel<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>>>>>>>>;
/*    let mut gemm2 = PartK<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U256,
                PartM<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U96,
                PartN<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U96,
                PartK<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U64,
                PartM<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U8,
                PartN<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>, U4,
                Ukernel<f64, HierA2<f64>, HierB2<f64>, HierC2<f64>>>>>>>>::new();*/
    let mut gemm2 : Blah = Blah::new();

    for index in 1..64 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = (index + 1) * 8;
        let (m, n, k) = (32, 24, 1);

        let n_reps = 1;
        for _ in 0..n_reps {
            let mut a : HierA2<f64> = Hierarch::new(m, k);
            let mut b : HierB2<f64> = Hierarch::new(k, n);
            let mut c : HierC2<f64> = Hierarch::new(m, n);
            a.fill_rand();
            b.fill_rand();
            c.fill_zero();

            let start = Instant::now();
            unsafe{
                gemm2.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
//                ref_gemm.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            
            let mut dur = start.elapsed();
            let time_secs = dur.as_secs() as f64;
            let time_nanos = dur.subsec_nanos() as f64;
            let time = time_nanos / 1E9 + time_secs;
            best_time = best_time.min(time);

            let err = test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err = worst_err.max(err);

            let mut a2 : Matrix<f64> = Matrix::new(m, k);
            let mut b2 : Matrix<f64> = Matrix::new(k, n);
            let mut c2 : Matrix<f64> = Matrix::new(m, n);
            a2.fill_rand(); b2.fill_rand(); c2.fill_zero();

            let start_blis = Instant::now();
            blas_dgemm( &mut a2, &mut b2, &mut c2);
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
    test_hierarch();
//    time_sweep_goto( );
}
