#![feature(specialization)]
#![feature(zero_one)]
#![feature(asm)]
#![feature(unique, alloc, heap_api)]

extern crate core;
use core::marker::{PhantomData};

use std::time::{Duration,Instant};

mod matrix;
mod gemm;
mod pack;
mod ukernel;
pub use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
pub use gemm::{GemmNode,PartM,PartN,PartK,PackArp,PackAcp,PackBrp,PackBcp,TripleLoopKernel};
pub use ukernel::{Ukernel};

extern crate libc;
use self::libc::{ c_double, int64_t, int32_t, c_char };
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

fn time_sweep_goto() -> ()
{

    let ukernel = Ukernel::new( 8, 4 );
//    let ukernel = TripleLoopKernel::new();
    let loop1: PartM<f64, RowPanelMatrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _> 
        = PartM::new( 8, 8, ukernel);
    let loop2: PartN<f64, RowPanelMatrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _> 
        = PartN::new( 4, 4, loop1 );
    let packa: PackArp<f64, Matrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _>
        = PackArp::new( 8, loop2 );
    let loop3: PartM<f64, Matrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _>
        = PartM::new( 96, 8, packa );
    let packb: PackBcp<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PackBcp::new( 4, loop3 );
    let loop4: PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartK::new( 256, 1, packb );
    let mut loop5: PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartN::new( 4096, 4, loop4 );

    for index in 0..32 {
        let mut best_time = 9999999999.0 as f64;
        let mut best_time_blis = 9999999999.0 as f64;
        let size = (index + 1) * 64;
        let m = size;
        let n = size;
        let k = 256;

        for _nrep in 0..5 {

            let mut a : Matrix<f64> = Matrix::new(m, k);
            let mut b : Matrix<f64> = Matrix::new(k, n);
            let mut c : Matrix<f64> = Matrix::new(m, n);
            a.fill_rand();
            b.fill_rand();
            c.fill_zero();
            
            let start = Instant::now();
            loop5.run( &mut a, &mut b, &mut c );
            let mut dur = start.elapsed();
            let time_secs = dur.as_secs() as f64;
            let time_nanos = dur.subsec_nanos() as f64;
            let time = time_nanos / 1E9 + time_secs;
            best_time = best_time.min(time);

            let start_blis = Instant::now();
            blas_dgemm( &mut a, &mut b, &mut c);
            dur = start_blis.elapsed();
            let time_secs_blis = dur.as_secs() as f64;
            let time_nanos_blis = dur.subsec_nanos() as f64;
            let time_blis = time_nanos_blis / 1E9 + time_secs_blis;
            best_time_blis = best_time_blis.min(time_blis);
        }
        let nflops = (m * n * k) as f64;
        println!("{}\t{}\t{}", size, 
                 2.0 * nflops / best_time / 1E9, 
                 2.0 * nflops / best_time_blis / 1E9);
    }
}


fn goto( a : &mut Matrix<f64>, b : &mut Matrix<f64>, c : &mut Matrix<f64> )
{
    let ukernel = Ukernel::new( 8, 4 );
    let loop1: PartM<f64, RowPanelMatrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _> 
        = PartM::new( 8, 8, ukernel);
    let loop2: PartN<f64, RowPanelMatrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _> 
        = PartN::new( 4, 4, loop1 );
    let packa: PackArp<f64, Matrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _>
        = PackArp::new( 8, loop2 );
    let loop3: PartM<f64, Matrix<f64>, ColumnPanelMatrix<f64>, Matrix<f64>, _>
        = PartM::new( 96, 8, packa );
    let packb: PackBcp<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PackBcp::new( 4, loop3 );
    let loop4: PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartK::new( 256, 1, packb );
    let mut loop5: PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _>
        = PartN::new( 4096, 4, loop4 );

    loop5.run( a, b, c );
}

fn main() {
    let m = 8;
    let n = 4;
    let k = 32;

    /*let mut t: Matrix<f64> = Matrix::new(m, m);
    let mut t_pack : RowPanelMatrix<f64> = RowPanelMatrix::new( t.height(), t.width(), 2 );
    t.fill_rand();
    t.print();
    println!("");
    t_pack.copy_from( &t );
    t_pack.print();
    println!("");
    return;*/




    let mut a : Matrix<f64> = Matrix::new(m, k);
    let mut b : Matrix<f64> = Matrix::new(k, n);
    let mut c : Matrix<f64> = Matrix::new(m, n);
    a.fill_rand();
    b.fill_rand();
    //c.fill_zero();
    c.fill_rand();


    let mut w : Matrix<f64> = Matrix::new(n, 1);
    let mut bw : Matrix<f64> = Matrix::new(k, 1);
    let mut abw : Matrix<f64> = Matrix::new(m, 1);
    let mut cw : Matrix<f64> = Matrix::new(m, 1);
    w.fill_rand();
    cw.fill_zero();
    bw.fill_zero();
    abw.fill_zero();


    //You can be explicit with the types like here:
    //The problem is that the types grow quadratically with the number of steps in the control tree!
/*    let kernel : TripleLoopKernel = TripleLoopKernel{};
    let loop1 : PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel> = PartM::new(5, 1, kernel);
    let loop2 : PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>> = PartK::new(4, 1, loop1);
    let mut loop3 : PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>>> = PartN::new(5, 1, loop2);*/


    //This one works as well. Can use new and types don't grow quadraticaly
/*    let kernelA : TripleLoopKernel = TripleLoopKernel{};
    let loopA : PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _> = PartM::new(5, 1, kernelA);
    let loopB : PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _> = PartK::new(4, 1, loopA);
    let mut loopC : PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, _> = PartN::new(5, 1, loopB); */
    
    //If you instead delcare the phantom data here, the type inference system works
    //I.E. the compiler can infer the type of loop1_inf when you use that type in loop2_inf
    //Now the code grows linearly with the number of steps in our control tree
    //The problem with this is that we have all this ugly phantom data laying around!
/*    let mut kernel_inf = TripleLoopKernel{};
    let mut loop1_inf = PartM{ bsz: 5, iota: 1, child: kernel_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let mut loop2_inf = PartK{ bsz: 5, iota: 1, child: loop1_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let mut loop3_inf = PartN{ bsz: 5, iota: 1, child: loop2_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let mut gemm = PackAcp{ panel_width: 5, child: loop3_inf,
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
*/
   a.print();
   println!("");
   b.print();
   println!("");
   c.print();
   println!("");
    let mut ref_gemm : TripleLoopKernel = TripleLoopKernel{};
    //C = AB
    //gemm.run( &mut a, &mut b, &mut c );
    //goto( &mut a, &mut b, &mut c );
    blas_dgemm( &mut a, &mut b, &mut c );

   a.print();
   println!("");
   b.print();
   println!("");
   c.print();
   println!("");
   /* a.print_wolfram();
    println!("");
    b.print_wolfram();
    println!("");
    c.print_wolfram();
    println!("");*/

    //Do bw = Bw, then abw = A*(Bw)   
    ref_gemm.run( &mut b, &mut w, &mut bw );
    ref_gemm.run( &mut a, &mut bw, &mut abw );

    //Do cw = Cw
    ref_gemm.run( &mut c, &mut w, &mut cw );
    
    //Cw -= abw
    cw.axpy( -1.0, &abw );
    let norm = cw.frosqr();
    println!("diff is {}\n", format!("{:e}",norm));

    time_sweep_goto( );
}
