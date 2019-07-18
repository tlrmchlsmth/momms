#![feature(specialization)]
#![feature(asm)] 
#![feature(const_generics)]

#![allow(unused_imports)]

extern crate core;
extern crate momms;

use std::time::{Instant};

use momms::kern::KernelNM;
use momms::matrix::{Scalar, Mat, ColumnPanelMatrix, RowPanelMatrix, Matrix, Hierarch};
use momms::composables::{GemmNode, AlgorithmStep, PartM, PartN, PartK, PackA, PackB, SpawnThreads, ParallelM, ParallelN, TheRest};
use momms::thread_comm::ThreadInfo;
use momms::util;
use core::marker::PhantomData;
/*
fn test_blas_dgemm ( m:usize, n: usize, k: usize, flusher: &mut Vec<f64>, n_reps: usize ) -> (f64, f64) 
{
    let mut best_time: f64 = 9999999999.0;
    let mut worst_err: f64 = 0.0;

    for _ in 0..n_reps {
        //Create matrices.
        let mut a : Matrix<f64> = Matrix::new(m, k);
        let mut b : Matrix<f64> = Matrix::new(k, n);
        let mut c : Matrix<f64> = Matrix::new(m, n);

        //Fill the matrices
        a.fill_rand(); c.fill_zero(); b.fill_rand();

        //Read a buffer so that A, B, and C are cold in cache.
        for i in flusher.iter_mut() { *i += 1.0; }
            
        //Time and run algorithm
        let start = Instant::now();
        util::blas_dgemm( &mut a, &mut b, &mut c);
        best_time = best_time.min(util::dur_seconds(start));
        let err = util::test_c_eq_a_b( &mut a, &mut b, &mut c);
        worst_err = worst_err.max(err);
    }
    (best_time, worst_err)
}*/


pub struct MMMTester<T: Scalar, const MR: usize, const NR: usize, const KC: usize> {
    _at: PhantomData<T>
}
impl<T: Scalar, const MR: usize, const NR: usize, const KC: usize> MMMTester<T, {MR}, {NR}, {KC}> {
    fn test_algorithm<S: GemmNode<T, Hierarch<T, {MR}, {KC}, {1}, {MR}>, Hierarch<T, {KC}, {NR}, {NR}, {1}>, Hierarch<T, {MR}, {NR}, {NR}, {1}>>>
        ( &self, m:usize, n: usize, k: usize, algo: &mut S, flusher: &mut Vec<f64>, n_reps: usize ) -> (f64, T) 
    {
        let algo_desc = S::hierarchy_description();
        let mut best_time: f64 = 9999999999.0;
        let mut worst_err: T = T::zero();

        for _ in 0..n_reps {
            //Create matrices.
            let mut a : Hierarch<T, {MR}, {KC}, {1}, {MR}> = Hierarch::new(m, k, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::K{bsz: 0});
            let mut b : Hierarch<T, {KC}, {NR}, {NR}, {1}> = Hierarch::new(k, n, &algo_desc, AlgorithmStep::K{bsz: 0}, AlgorithmStep::N{bsz: 0});
            let mut c : Hierarch<T, {MR}, {NR}, {NR}, {1}> = Hierarch::new(m, n, &algo_desc, AlgorithmStep::M{bsz: 0}, AlgorithmStep::N{bsz: 0});

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
}

fn test() {
    type GotoA<T> = Hierarch<T,   4, 192,  1, 4>;
    type GotoB<T> = Hierarch<T, 192,  12, 12, 1>;
    type GotoC<T> = Hierarch<T,   4,  12, 12, 1>;
    type Goto<T,MTA,MTB,MTC> 
        = SpawnThreads<T, MTA, MTB, MTC,
            PartN<T, MTA, MTB, MTC,
                PartK<T, MTA, MTB, MTC,
                    PartM<T, MTA, MTB, MTC,
                        ParallelN<T, MTA, MTB, MTC, TheRest, KernelNM<T, MTA, MTB, MTC, 4, 12>, 12>, 
                    120>,
                192>,
            3000>
          >;

    type L3bA<T> = Hierarch<T,   4, 192, 1,  4>;
    type L3bB<T> = Hierarch<T, 192,  12, 12, 1>;
    type L3bC<T> = Hierarch<T,   4,  12, 12, 1>;
    type L3B<T,MTA,MTB,MTC> 
        = SpawnThreads<T, MTA, MTB, MTC,
              PartN<T, MTA, MTB, MTC,
                  PartK<T, MTA, MTB, MTC,
                      PartM<T, MTA, MTB, MTC,
                          PartK<T, MTA, MTB, MTC,
                              ParallelN<T, MTA, MTB, MTC, TheRest, KernelNM<T, MTA, MTB, MTC, 12, 4>, 12>,
                          192>,
                      120>,
                  768>,
              768>,
          >;

    let mut goto = <Goto<f64, GotoA<f64>, GotoB<f64>, GotoC<f64>>>::new();
    let mut l3b = <L3B<f64, L3bA<f64>, L3bB<f64>, L3bC<f64>>>::new();
    goto.set_n_threads(4);
    l3b.set_n_threads(4);

    //Initialize array to flush cache with
    let flusher_len = 2*1024*1024; //16MB
    let mut flusher : Vec<f64> = Vec::with_capacity(flusher_len);
    for _ in 0..flusher_len { flusher.push(0.0); }

    println!("m\tn\tk\t{: <13}{: <13}{: <15}{: <15}", "goto", "l3b", "goto", "l3b");
    for index in 01..81 {
        let size = index*50;
        let (m, n, k) = (size, size, size);

        let n_reps = 5;
        
       // let tman : MMMTester<f64, 4, 12, 192> = { _at : PhantomData };
        //let tman = MMMTester<f64, 4, 12, 192> {};
        let tman : MMMTester<f64, 4, 12, 192> = MMMTester {_at : PhantomData};
        let (goto_time, goto_err) = tman.test_algorithm(m, n, k, &mut goto, &mut flusher, n_reps);
        let (l3b_time, l3b_err) = tman.test_algorithm(m, n, k, &mut l3b, &mut flusher, n_reps);
        //let (goto_time, goto_err) = MMMTester<f64, {4usize, 12usize, 192usize}>::test_algorithm(m, n, k, &mut goto, &mut flusher, n_reps);
//        let (goto_time, goto_err) = MMMTester<f64, {4}, {12}, {192}>::test_algorithm(m, n, k, &mut goto, &mut flusher, n_reps);

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
