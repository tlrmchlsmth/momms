#![feature(specialization)]
#![feature(asm)] 
#![allow(unused_imports)]

extern crate core;
extern crate typenum;
extern crate momms;

use std::time::{Instant};
use typenum::{Unsigned,U1};

use momms::kern::KnmKernel;
use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix, Hierarch};
use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB, UnpackC, SpawnThreads, ParallelM, ParallelN, TheRest, Target};
use momms::thread_comm::ThreadInfo;
use momms::util;

fn test_algorithm_flat<T: Scalar, S: GemmNode<T, Matrix<T>, Matrix<T>, Matrix<T>>>
    ( m:usize, n: usize, k: usize, algo: &mut S, flusher: &mut Vec<f64>, n_reps: usize ) -> (f64, T) 
{
    let mut best_time: f64 = 9999999999.0;
    let mut worst_err: T = T::zero();

    for _ in 0..n_reps {
        //Create matrices.
        let mut a = <Matrix<T>>::new(m, k);
        let mut b = <Matrix<T>>::new(k, n);
        let mut c = <Matrix<T>>::new(m, n);

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
    
fn test_algorithm<T: Scalar, Mr: Unsigned, Nr: Unsigned, Kr:Unsigned, 
    S: GemmNode<T, Hierarch<T, Mr, Kr, U1, Mr>, Hierarch<T, Kr, Nr, U1, Kr>, Hierarch<T, Mr, Nr, U1, Mr>>>
    ( m:usize, n: usize, k: usize, algo: &mut S, flusher: &mut Vec<f64>, n_reps: usize ) -> (f64, T) 
{
    let algo_desc = S::hierarchy_description();
    let mut best_time: f64 = 9999999999.0;
    let mut worst_err: T = T::zero();

    for _ in 0..n_reps {
        //Create matrices.
        let mut a : Hierarch<T, Mr, Kr, U1, Mr> = Hierarch::new(m, k, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
        let mut b : Hierarch<T, Kr, Nr, U1, Kr> = Hierarch::new(k, n, &algo_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
        let mut c : Hierarch<T, Mr, Nr, U1, Mr> = Hierarch::new(m, n, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});

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
    use typenum::{UInt, UTerm, B0, B1, Unsigned};
    
    type U14400 = UInt<UInt<UInt<UInt<typenum::U900, B0>, B0>, B0>, B0>;
    type NC = U14400;
    type KC = typenum::U336;
    type MC = typenum::U160; 
    type MR = typenum::U16;
    type NR = typenum::U24;
    type KR = typenum::U4;
    
//    type JCWAY = Target<typenum::U4>;
   
    type Algo<T, MTA, MTB, MTC>
        = PartN<T, MTA, MTB, MTC, NC,
          PartK<T, MTA, MTB, MTC, KC,
          PackB<T, MTA, MTB, MTC, Hierarch<T, KR, NR, U1, KR>,
          PartM<T, MTA, Hierarch<T, KR, NR, U1, KR>, MTC, MC,
          PackA<T, MTA, Hierarch<T, KR, NR, U1, KR>, MTC, Hierarch<T, MR, KR, U1, MR>,
          PartN<T, Hierarch<T, MR, KR, U1, MR>, Hierarch<T, KR, NR, U1, KR>, MTC, NR,
          PartM<T, Hierarch<T, MR, KR, U1, MR>, Hierarch<T, KR, NR, U1, KR>, MTC, MR,
          KnmKernel<T, Hierarch<T, MR, KR, U1, MR>, Hierarch<T, KR, NR, U1, KR>, MTC, MR, NR>>>>>>>>;

    type AlgoNopack<T, MTA, MTB, MTC>
        = PartN<T, MTA, MTB, MTC, NC,
          PartK<T, MTA, MTB, MTC, KC,
          PartM<T, MTA, MTB, MTC, MC,
          PartN<T, MTA, MTB, MTC, NR,
          PartM<T, MTA, MTB, MTC, MR,
          KnmKernel<T, MTA, MTB, MTC, MR, NR>>>>>>;

    let mut algo = <Algo<f32, Matrix<f32>, Matrix<f32>, Matrix<f32>>>::new();
    let mut algo_nopack = <AlgoNopack<f32, Hierarch<f32, MR, KR, U1, MR>, Hierarch<f32, KR, NR, U1, KR>, Hierarch<f32, MR, NR, U1, MR>>>::new();

    //Initialize array to flush cache with
    let flusher_len = 2*1024*1024; //16MB
    let mut flusher : Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len { flusher.push(0.0); }

    println!("m\tn\tk\t{: <13}{: <13}{: <15}{: <15}", "pack", "no pack", "pack", "no pack");
    for index in 1..100 {
        let size = 2*48*index;
        let (m, n, k) = (size, size, size);

        let n_reps = 1;
        let (goto_time, goto_err) = test_algorithm_flat(m, n, k, &mut algo, &mut flusher, n_reps);
        let (l3b_time, l3b_err) = test_algorithm(m, n, k, &mut algo_nopack, &mut flusher, n_reps);

        println!("{}\t{}\t{}\t{}{}{}{}", 
                 m, n, k,
                 format!("{: <13.5}", util::gflops(m,n,k,goto_time)), 
                 format!("{: <13.5}", util::gflops(m,n,k,l3b_time)), 
                 format!("{: <15.5e}", goto_err.sqrt()),
                 format!("{: <15.5e}", l3b_err.sqrt()));

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
