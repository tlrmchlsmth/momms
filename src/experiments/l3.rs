#![feature(specialization)]
#![feature(conservative_impl_trait)]
#![feature(cfg_target_feature)]
#![feature(asm)] 

#![allow(unused_imports)]

extern crate core;
extern crate typenum;
extern crate mommies;

use std::time::{Instant};
use typenum::{U1};

use mommies::kern::hsw::KernelNM;
use mommies::matrix::{Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix, Hierarch};
use mommies::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB, SpawnThreads, ParallelM, ParallelN, TheRest};
use mommies::thread_comm::ThreadInfo;
use mommies::util;


fn flush_cache( arr: &mut Vec<f64> ) {
    for i in (arr).iter_mut() {
        *i += 1.0;
    }

}

fn test() {
    use typenum::{UInt, UTerm, B0, B1, Unsigned};
    type U3000 = UInt<UInt<typenum::U750, B0>, B0>;
    type NC = U3000;
    type KC = typenum::U192; 
    type MC = typenum::U120; 
    type MR = typenum::U4;
    type NR = typenum::U12;
    type GotoA<T> = Hierarch<T, MR, KC, U1, MR>;
    type GotoB<T> = Hierarch<T, KC, NR, NR, U1>;
    type GotoC<T> = Hierarch<T, MR, NR, NR, U1>;

    type NcL3 = typenum::U768;
    type KcL3 = typenum::U768;
    type McL2 = typenum::U120;
    type KcL2 = typenum::U192;
    type L3bA<T> = Hierarch<T, MR, KcL2, U1, MR>;
    type L3bB<T> = Hierarch<T, KcL2, NR, NR, U1>;
    type L3bC<T> = Hierarch<T, MR, NR, NR, U1>;

    type GotoH<T,MTA,MTB,MTC> 
        = SpawnThreads<T, MTA, MTB, MTC,
          PartN<T, MTA, MTB, MTC, NC,
          PartK<T, MTA, MTB, MTC, KC,
          PartM<T, MTA, MTB, MTC, MC,
          ParallelN<T, MTA, MTB, MTC, NR, TheRest,
          KernelNM<T, MTA, MTB, MTC, NR, MR>>>>>>;

    type L3H<T,MTA,MTB,MTC> 
        = SpawnThreads<T, MTA, MTB, MTC,
          PartN<T, MTA, MTB, MTC, NcL3,
          PartK<T, MTA, MTB, MTC, KcL3,
          PartM<T, MTA, MTB, MTC, McL2,
          //Barrier<T, MTA, MTB, MTC,
          PartK<T, MTA, MTB, MTC, KcL2,
          ParallelN<T, MTA, MTB, MTC, NR, TheRest,
          KernelNM<T, MTA, MTB, MTC, NR, MR>>>>>>>;

    type GotoHier = GotoH<f64, GotoA<f64>, GotoB<f64>, GotoC<f64>>;
    type L3Hier = L3H<f64, L3bA<f64>, L3bB<f64>, L3bC<f64>>;

    let mut goto_hier : GotoHier = GotoHier::new();
    let mut l3_hier : L3Hier = L3Hier::new();
    goto_hier.set_n_threads(4);
    l3_hier.set_n_threads(4);
    let goto_desc = GotoHier::hierarchy_description();
    let l3_desc = L3Hier::hierarchy_description();

    let flusher_len = 2*1024*1024; //16MB
    let mut flusher : Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

//    pin_to_core(0);

    for index in 01..50 {
        let mut best_time_goto: f64 = 9999999999.0;
        let mut best_time_l3: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let mut worst_err_l3: f64 = 0.0;
        let size = index*64;
        let (m, n, k) = (size, size, size);


        let n_reps = 6;
        for _ in 0..n_reps {
            let mut a : GotoA<f64> = Hierarch::new(m, k, &goto_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b : GotoB<f64> = Hierarch::new(k, n, &goto_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c : GotoC<f64> = Hierarch::new(m, n, &goto_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});
            a.fill_rand(); c.fill_zero(); b.fill_rand();

            flush_cache(&mut flusher);
            let mut start = Instant::now();
            unsafe{
                goto_hier.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() );
            }
            best_time_goto = best_time_goto.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b( &mut a, &mut b, &mut c);
            worst_err = worst_err.max(err);

            let mut a3 : Matrix<f64> = Matrix::new(m, k);
            let mut b3 : Matrix<f64> = Matrix::new(k, n);
            let mut c3 : Matrix<f64> = Matrix::new(m, n);
            a3.fill_rand(); c3.fill_zero(); b3.fill_rand();
            
            flush_cache(&mut flusher);
            start = Instant::now();
            util::blas_dgemm( &mut a3, &mut b3, &mut c3);
            best_time_blis = best_time_blis.min(util::dur_seconds(start));

            let mut a2 : L3bA<f64> = Hierarch::new(m, k, &l3_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b2 : L3bB<f64> = Hierarch::new(k, n, &l3_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c2 : L3bC<f64> = Hierarch::new(m, n, &l3_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});
            a2.fill_rand(); c2.fill_zero(); b2.fill_rand();

            flush_cache(&mut flusher);
            start = Instant::now();
            unsafe{
                l3_hier.run( &mut a2, &mut b2, &mut c2, &ThreadInfo::single_thread() );
            }
            best_time_l3 = best_time_l3.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b( &mut a2, &mut b2, &mut c2);
            worst_err_l3 = worst_err_l3.max(err);
           
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", 
                 m, n, k,
                 util::gflops(m,n,k,best_time_goto), 
                 util::gflops(m,n,k,best_time_l3), 
                 util::gflops(m,n,k,best_time_blis),
                 format!("{:e}", worst_err.sqrt()),
                 format!("{:e}", worst_err_l3.sqrt()));

    }

    let mut sum = 0.0;
    for a in flusher.iter() {
        sum += *a;
    }
    println!("Flush value {}", sum);
}

fn main() {
//    test_gemv_kernel();
//    compare_gotos( );
    test( );
}
