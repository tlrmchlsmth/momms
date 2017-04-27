#![feature(specialization)]
#![feature(conservative_impl_trait)]
#![feature(cfg_target_feature)]
#![feature(asm)] 

#![allow(unused_imports)]

extern crate typenum;
extern crate momms;

use std::time::{Instant};
use typenum::{U1};

use momms::kern::hsw::KernelNM;
use momms::matrix::{Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix, Hierarch};
use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB, SpawnThreads, ParallelM, ParallelN, TheRest};
use momms::thread_comm::ThreadInfo;
use momms::util;

fn flush_cache( arr: &mut Vec<f64> ) {
    for i in (arr).iter_mut() {
        *i += 1.0;
    }

}

fn test() {
    use typenum::{UInt, B0};
    type U3000 = UInt<UInt<typenum::U750, B0>, B0>;
    type Nc = U3000;
    type Kc = typenum::U192; 
    type Mc = typenum::U120; 
    type Mr = typenum::U4;
    type Nr = typenum::U12;
    type GotoA<T> = Hierarch<T, Mr, Kc, U1, Mr>;
    type GotoB<T> = Hierarch<T, Kc, Nr, Nr, U1>;
    type GotoC<T> = Hierarch<T, Mr, Nr, Nr, U1>;

    type GotoH<T,MTA,MTB,MTC> 
        = SpawnThreads<T, MTA, MTB, MTC,
          PartN<T, MTA, MTB, MTC, Nc,
          PartK<T, MTA, MTB, MTC, Kc,
          PartM<T, MTA, MTB, MTC, Mc,
          ParallelN<T, MTA, MTB, MTC, Nr, TheRest,
          KernelNM<T, MTA, MTB, MTC, Nr, Mr>>>>>>;

    type U3600 = UInt<UInt<typenum::U900, B0>, B0>;
    type NcL4 = U3600;
    type McL4 = U3600;
    type L4C<T,MTA,MTB,MTC> 
        = SpawnThreads<T, MTA, MTB, MTC,
          PartM<T, MTA, MTB, MTC, McL4,
          PartN<T, MTA, MTB, MTC, NcL4,
          PartK<T, MTA, MTB, MTC, Kc,
          PartM<T, MTA, MTB, MTC, Mc,
          ParallelN<T, MTA, MTB, MTC, Nr, TheRest,
          KernelNM<T, MTA, MTB, MTC, Nr, Mr>>>>>>>;

    type GotoHier = GotoH<f64, GotoA<f64>, GotoB<f64>, GotoC<f64>>;
    type L4Hier = L4C<f64, GotoA<f64>, GotoB<f64>, GotoC<f64>>;

    let mut goto_hier : GotoHier = GotoHier::new();
    let mut l4_hier : L4Hier = L4Hier::new();
    goto_hier.set_n_threads(4);
    l4_hier.set_n_threads(4);

    let goto_desc = GotoHier::hierarchy_description();
    let l4_desc = L4Hier::hierarchy_description();

    let flusher_len = 32*1024*1024; //256MB
    let mut flusher : Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

//    pin_to_core(0);

    for index in 01..100 {
        let mut best_time_goto: f64 = 9999999999.0;
        let mut best_time_l3: f64 = 9999999999.0;
        let mut best_time_blis: f64 = 9999999999.0;
        let mut worst_err: f64 = 0.0;
        let mut worst_err_l3: f64 = 0.0;
        let size = index*512;
        let (m, n, k) = (size, size, size);


        let n_reps = 5;
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
            //blas_dgemm( &mut a3, &mut b3, &mut c3);
            best_time_blis = best_time_blis.min(util::dur_seconds(start));

    /*        let mut a2 : L3bA<f64> = Hierarch::new(m, k, &l3_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b2 : L3bB<f64> = Hierarch::new(k, n, &l3_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c2 : L3bC<f64> = Hierarch::new(m, n, &l3_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});*/
            let mut a2 : GotoA<f64> = Hierarch::new(m, k, &l4_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b2 : GotoB<f64> = Hierarch::new(k, n, &l4_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c2 : GotoC<f64> = Hierarch::new(m, n, &l4_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});
            a2.fill_rand(); c2.fill_zero(); b2.fill_rand();

            flush_cache(&mut flusher);
            start = Instant::now();
            unsafe{
                l4_hier.run( &mut a2, &mut b2, &mut c2, &ThreadInfo::single_thread() );
            }
            best_time_l3 = best_time_l3.min(util::dur_seconds(start));
            let err = util::test_c_eq_a_b( &mut a2, &mut b2, &mut c2);
            worst_err_l3 = worst_err_l3.max(err);
           
        }
        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}", 
                 m, n, k,
                 util::gflops(m,n,k,best_time_goto), 
                 util::gflops(m,n,k,best_time_l3), 
        //         util::gflops(m,n,k,best_time_blis),
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
