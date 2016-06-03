#![feature(specialization)]
#![feature(zero_one)]
extern crate core;

mod matrix;
mod cntl;
use core::marker::{PhantomData};
pub use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix};
pub use cntl::{GemmNode,PartM,PartN,PartK,PackArp,PackAcp,PackBrp,PackBcp,TripleLoopKernel};







fn main() {
    let m = 10;
    let n = 10;
    let k = 10;

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
    gemm.run( &mut a, &mut b, &mut c );
    //loop3.run( &mut a, &mut b, &mut c );

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
