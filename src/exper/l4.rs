#![feature(specialization)]
#![feature(conservative_impl_trait)]
#![feature(cfg_target_feature)]
#![feature(asm)] 

#![allow(unused_imports)]

extern crate typenum;
extern crate momms;

use std::time::{Instant};
use typenum::{Unsigned,U1};

use momms::kern::KernelNM;
use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix, Hierarch};
use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB, SpawnThreads, ParallelM, ParallelN, TheRest};
use momms::thread_comm::ThreadInfo;
use momms::util;

fn test_algorithm<T: Scalar, Mr: Unsigned, Nr: Unsigned, Kc:Unsigned, CLRS: Unsigned, CLCS: Unsigned, 
    S: GemmNode<T, Hierarch<T, Mr, Kc, U1, Mr>, Hierarch<T, Kc, Nr, Nr, U1>, Hierarch<T, Mr, Nr, CLRS, CLCS>>>
    ( m:usize, n: usize, k: usize, algo: &mut S, flusher: &mut Vec<f64>, n_reps: usize ) -> (f64, T) 
{
    let algo_desc = S::hierarchy_description();
    let mut best_time: f64 = 9999999999.0;
    let mut worst_err: T = T::zero();

    for _ in 0..n_reps {
        //Create matrices.
        let mut a : Hierarch<T, Mr, Kc, U1, Mr> = Hierarch::new(m, k, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
        let mut b : Hierarch<T, Kc, Nr, Nr, U1> = Hierarch::new(k, n, &algo_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
        let mut c : Hierarch<T, Mr, Nr, CLRS, CLCS> = Hierarch::new(m, n, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});

        //Fill the matrices
        a.fill_rand(); c.fill_zero(); b.fill_rand();

        //Read a buffer so that A, B, and C are cold in cache.
        for i in flusher.iter_mut() { *i += 1.0; }

        //Time and run algorithm
        let start = Instant::now();
        unsafe{ algo.run( &mut a, &mut b, &mut c, &ThreadInfo::single_thread() ); }
        best_time = best_time.min(util::dur_seconds(start));
        let err = util::test_c_eq_a_b( &mut a, &mut b, &mut c);
        worst_err = worst_err.max(err);
    }
    (best_time, worst_err)
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

    type Goto<T,MTA,MTB,MTC> 
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

    let mut goto = <Goto<f64, GotoA<f64>, GotoB<f64>, GotoC<f64>>>::new();
    let mut l4c = <L4C<f64, GotoA<f64>, GotoB<f64>, GotoC<f64>>>::new();
    goto.set_n_threads(4);
    l4c.set_n_threads(4);

    let flusher_len = 32*1024*1024; //256MB
    let mut flusher : Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len {
        flusher.push(0.0);
    }

    for index in 01..100 {
        let size = index*512;
        let (m, n, k) = (size, size, size);

        let n_reps = 6;
        let (goto_time, goto_err) = test_algorithm(m, n, k, &mut goto, &mut flusher, n_reps);
        let (l4c_time, l4c_err) = test_algorithm(m, n, k, &mut l4c, &mut flusher, n_reps);

        println!("{}\t{}\t{}\t{}\t{}\t{}\t{}", 
                 m, n, k,
                 format!("{:5.5}", util::gflops(m,n,k,goto_time)), 
                 format!("{:5.5}", util::gflops(m,n,k,l4c_time)), 
                 format!("{:5.5e}", goto_err.sqrt()),
                 format!("{:5.5e}", l4c_err.sqrt()));
    }

    let mut sum = 0.0;
    for a in flusher.iter() {
        sum += *a;
    }
    println!("Flush value {}", sum);
}

fn main() {
    test( );
}
