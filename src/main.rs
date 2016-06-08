#![feature(specialization)]
#![feature(zero_one)]
#![feature(asm)]

extern crate core;
use core::marker::{PhantomData};

use std::time::{Duration,Instant};

mod matrix;
pub use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
mod cntl;
pub use cntl::{GemmNode,PartM,PartN,PartK,PackArp,PackAcp,PackBrp,PackBcp,TripleLoopKernel};
mod ukernel;
pub use ukernel::{Ukernel};



fn time_sweep_goto() -> ()
{
    let ukernel = Ukernel{ mr: 8, nr: 4, _t: PhantomData::<f64> };
    let loop1 = PartM{ bsz: 8, iota: 8, child: ukernel, 
        _t: PhantomData::<f64>, _at: PhantomData::<RowPanelMatrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop2 = PartN{ bsz: 4, iota: 4, child: loop1, 
        _t: PhantomData::<f64>, _at: PhantomData::<RowPanelMatrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let packa = PackArp{ panel_height: 8, child: loop2, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop3 = PartM{ bsz: 96, iota: 8, child: packa, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let packb = PackBcp{ panel_width: 4, child: loop3, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop4 = PartK{ bsz: 256, iota: 1, child: packb, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop5 = PartN{ bsz: 4096, iota: 4, child: loop4, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };

    for index in 0..64 {
        let size = (index + 1) * 16;
        let mut a : Matrix<f64> = Matrix::new(size, size);
        let mut b : Matrix<f64> = Matrix::new(size, size);
        let mut c : Matrix<f64> = Matrix::new(size, size);
        a.fill_rand();
        b.fill_rand();
        c.fill_zero();
        
        let start = Instant::now();
        loop5.run( &mut a, &mut b, &mut c );
        let time = start.elapsed().subsec_nanos() as f64;

        let nflops = (2 * size * size * size) as f64;
        println!("{}\t{}", size, nflops / time);
        
    }
}


fn goto( a : &mut Matrix<f64>, b : &mut Matrix<f64>, c : &mut Matrix<f64> )
{
    let ukernel = Ukernel{ mr: 8, nr: 4, _t: PhantomData::<f64> };
    let loop1 = PartM{ bsz: 8, iota: 8, child: ukernel, 
        _t: PhantomData::<f64>, _at: PhantomData::<RowPanelMatrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop2 = PartN{ bsz: 4, iota: 4, child: loop1, 
        _t: PhantomData::<f64>, _at: PhantomData::<RowPanelMatrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let packa = PackArp{ panel_height: 8, child: loop2, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop3 = PartM{ bsz: 96, iota: 8, child: packa, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<ColumnPanelMatrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let packb = PackBcp{ panel_width: 4, child: loop3, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop4 = PartK{ bsz: 256, iota: 1, child: packb, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop5 = PartN{ bsz: 4096, iota: 4, child: loop4, 
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };

    loop5.run( a, b, c );
}

fn main() {
    let m = 16;
    let n = 16;
    let k = 16;

    /*let mut t: Matrix<f64> = Matrix::new(m, m);
    let mut t_pack : RowPanelMatrix<f64> = RowPanelMatrix::new( t.height(), t.width(), 2 );
    t.fill_rand();
    t.print();
    println!("");
    t_pack.copy_from( &t );
    t_pack.print();
    println!("");
    return;*/

    time_sweep_goto( );



    let mut a : Matrix<f64> = Matrix::new(m, k);
    let mut b : Matrix<f64> = Matrix::new(k, n);
    let mut c : Matrix<f64> = Matrix::new(m, n);
    a.fill_rand();
    b.fill_rand();
    c.fill_zero();


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
    let kernel : TripleLoopKernel = TripleLoopKernel{};
    let loop1 : PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel> = PartM::new(5, 1, kernel);
    let loop2 : PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>> = PartK::new(4, 1, loop1);
    let loop3 : PartN<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartK<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, Matrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>>> = PartN::new(5, 1, loop2);

    //If you instead delcare the phantom data here, the type inference system works
    //I.E. the compiler can infer the type of loop1_inf when you use that type in loop2_inf
    //Now the code grows linearly with the number of steps in our control tree
    //The problem with this is that we have all this ugly phantom data laying around!
    let kernel_inf = TripleLoopKernel{};
    let loop1_inf = PartM{ bsz: 5, iota: 1, child: kernel_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop2_inf = PartK{ bsz: 5, iota: 1, child: loop1_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop3_inf = PartN{ bsz: 5, iota: 1, child: loop2_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let gemm = PackAcp{ panel_width: 5, child: loop3_inf,
        _t: PhantomData::<f64>, _at: PhantomData::<Matrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };

    let ref_gemm : TripleLoopKernel = TripleLoopKernel{};
    //C = AB
    //gemm.run( &mut a, &mut b, &mut c );
    //loop3.run( &mut a, &mut b, &mut c );
    goto( &mut a, &mut b, &mut c );

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
    println!("diff is {}", norm);
}
