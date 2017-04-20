extern crate mommies;
extern crate typenum;

use std::time::{Instant};
use typenum::{U1};
use mommies::kern::hsw::{KernelNM, GemvAL1};
pub use mommies::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix, Hierarch};
pub use mommies::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB, SpawnThreads, ParallelM, ParallelN, Nwayer};
pub use mommies::thread_comm::ThreadInfo;
pub use mommies::util;

#[allow(dead_code)]
fn compare_gotos() {
    type KC = typenum::U256; 
    type MC = typenum::U72; 
    type NR = typenum::U8;
    type MR = typenum::U6;
    type HierA<T> = Hierarch<T, MR, KC, U1, MR>;
    type HierB<T> = Hierarch<T, KC, NR, NR, U1>;
    type HierC<T> = Hierarch<T, MR, NR, NR, U1>;

    type GotoH<T,MTA,MTB,MTC> 
        = PartK<T, MTA, MTB, MTC, KC,
          PartM<T, MTA, MTB, MTC, MC,
          KernelNM<T, MTA, MTB, MTC, NR, MR>>>;

    type Goto<T,MTA,MTB,MTC> 
        = PartK<T, MTA, MTB, MTC, KC,
          PackB<T, MTA, MTB, MTC, ColumnPanelMatrix<T,NR>,
          PartM<T, MTA, ColumnPanelMatrix<T,NR>, MTC, MC,
          PackA<T, MTA, ColumnPanelMatrix<T,NR>, MTC, RowPanelMatrix<T,MR>,
          KernelNM<T, RowPanelMatrix<T,MR>, ColumnPanelMatrix<T,NR>, MTC, NR, MR>>>>>;

    type GotoOrig  = Goto<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>>;
    type GotoHierC = Goto<f64, Matrix<f64>, Matrix<f64>, HierC<f64>>;
    type GotoHier = GotoH<f64, HierA<f64>, HierB<f64>, HierC<f64>>;

    let mut goto : GotoOrig = GotoOrig::new();
    let mut goto_hier_c : GotoHierC = GotoHierC::new();
    let mut goto_hier : GotoHier = GotoHier::new();
    let algo_desc = GotoHier::hierarchy_description();
    
    util::pin_to_core(0);

    for index in 1..128 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut best_time_2: f64 = 9999999999.0;
        let mut best_time_3: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let mut worst_err_2: f64 = 0.0;
        let mut worst_err_3: f64 = 0.0;
        let size = index*8;
        let (m, n, k) = (size, size, size);


        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a : HierA<f64> = Hierarch::new(m, k, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b : HierB<f64> = Hierarch::new(k, n, &algo_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c : HierC<f64> = Hierarch::new(m, n, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});
            a.fill_rand(); b.fill_rand(); c.fill_zero();

            let mut a2 : Matrix<f64> = Matrix::new(m, k);
            let mut b2 : Matrix<f64> = Matrix::new(k, n);
            let mut c2 : Matrix<f64> = Matrix::new(m, n);
            a2.fill_rand(); b2.fill_rand(); c2.fill_zero();

            c2.transpose();
            let mut start = Instant::now();
            unsafe{
                goto.run( &mut a2, &mut b2, &mut c2, &ThreadInfo::single_thread() );
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b( &mut a2, &mut b2, &mut c2);
            worst_err = worst_err.max(err);
            c2.transpose();

            start = Instant::now();
            unsafe{
                goto_hier_c.run( &mut a2, &mut b2, &mut c, &ThreadInfo::single_thread() );
            }
            best_time_2 = best_time_2.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b( &mut a2, &mut b2, &mut c);
            worst_err_2 = worst_err_2.max(err);
            
            c.fill_zero();           
            start = Instant::now();
            unsafe{
                goto_hier.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            best_time_3 = best_time_3.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err_3 = worst_err_3.max(err);
            

            start = Instant::now();
            util::blas_dgemm( &mut a2, &mut b2, &mut c2);
            best_time_blis = best_time_blis.min(util::dur_seconds(start));
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", 
                 m, n, k,
                 util::gflops(m,n,k,best_time), 
                 util::gflops(m,n,k,best_time_2), 
                 util::gflops(m,n,k,best_time_3), 
                 util::gflops(m,n,k,best_time_blis),
                 format!("{:e}", worst_err.sqrt()),
                 format!("{:e}", worst_err_2.sqrt()),
                 format!("{:e}", worst_err_3.sqrt()));

    }
}

fn test_gemv_kernel() {
    type KC = typenum::U64; 
    type MC = typenum::U56;
    type NC = typenum::U1024;

    type HierA<T> = Hierarch<T, MC, KC, U1, MC>;
    type HierB<T> = Hierarch<T, KC, NC, U1, KC>;
    type HierC<T> = Hierarch<T, MC, NC, U1, MC>;

    type Algorithm<T,MTA,MTB,MTC> 
        = PartK<T, MTA, MTB, MTC, typenum::U128,
          PartN<T, MTA, MTB, MTC, typenum::U192, 
          PartK<T, MTA, MTB, MTC, KC,
          PartM<T, MTA, MTB, MTC, MC,
          GemvAL1<T, MTA, MTB, MTC, MC, KC>>>>>;

    type Algo = Algorithm<f64, HierA<f64>, HierB<f64>, HierC<f64>>;

    let mut algo : Algo = Algo::new();
    let algo_desc = Algo::hierarchy_description();
    
    util::pin_to_core(0);

    for index in 1..128 {
        let mut best_time: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let size = index*8;
        let (m, n, k) = (size, size, size);

        let n_reps = 5;
        for _ in 0..n_reps {
            let mut a : HierA<f64> = Hierarch::new(m, k, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b : HierB<f64> = Hierarch::new(k, n, &algo_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c : HierC<f64> = Hierarch::new(m, n, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});
            a.fill_rand(); b.fill_rand(); c.fill_zero();

            let mut a2 : Matrix<f64> = Matrix::new(m, k);
            let mut b2 : Matrix<f64> = Matrix::new(k, n);
            let mut c2 : Matrix<f64> = Matrix::new(m, n);
            a2.fill_rand(); b2.fill_rand(); c2.fill_zero();

            let mut start = Instant::now();
            unsafe{
                algo.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            best_time = best_time.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err = worst_err.max(err);

/*
            print!("A = ");
            a.print_matlab();
            println!(";");
            println!("");
            print!("B = ");
            b.print_matlab();
            println!(";");
            println!("");
            print!("C = ");
            c.print_matlab();
            println!(";");
            println!("");
            
            println!("");
            c.print();
*/
            start = Instant::now();
            util::blas_dgemm( &mut a2, &mut b2, &mut c2);
            best_time_blis = best_time_blis.min(util::dur_seconds(start));
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}", 
                 m, n, k,
                 util::gflops(m,n,k,best_time), 
                 util::gflops(m,n,k,best_time_blis),
                 format!("{:e}", worst_err.sqrt()));
    }
}

fn main() {
    test_gemv_kernel();
//    time_sweep_goto( );
}
