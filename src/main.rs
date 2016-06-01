#![feature(specialization)]
#![feature(zero_one)]
extern crate core;
extern crate rand;

use std::ops::{Add, Mul, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::num::{Zero};
use core::fmt::{Display};
use core::marker::PhantomData;

trait Scalar where
    Self: Add<Self, Output=Self>,
    Self: Mul<Self, Output=Self>,
    Self: Sub<Self, Output=Self>,
    Self: Div<Self, Output=Self>,
    Self: AddAssign<Self>,
    Self: MulAssign<Self>,
    Self: SubAssign<Self>,
    Self: DivAssign<Self>,
    Self: Zero,
    Self: Sized,
    Self: Copy,
    Self: Display,
    Self: rand::Rand,
{}
impl Scalar for f64{}
impl Scalar for f32{}

struct Matrix<T: Scalar> {
    h: usize,
    w: usize,
    row_stride: usize,
    column_stride: usize,
    buffer: Vec<T>,
}
impl<T: Scalar> Matrix<T> {
    fn new( h: usize, w: usize ) -> Matrix<T> {
        let mut buf = Vec::with_capacity( h * w );
        unsafe{ buf.set_len( h * w );}
        let mat = Matrix{ h: h, w: w, row_stride: 1, column_stride: h, buffer: buf };
        return mat;
    }
    fn get( &self, y: usize, x: usize) -> T {
        return self.buffer[y*self.row_stride + x*self.column_stride];
    }
    fn set( &mut self, y: usize, x: usize, alpha: T) -> () {
        self.buffer[y*self.row_stride + x*self.column_stride] = alpha;
    }
    fn fill_rand( &mut self ) -> () {
        for x in 0..self.w {
            for y in 0..self.h {
                self.set(y,x,rand::random::<T>());
            }
        }
    }
    fn fill_zero( &mut self ) -> () {
        for x in 0..self.w {
            for y in 0..self.h {
                self.set(y,x,T::zero());
            }
        }
    }
    fn frosqr( &self ) -> T {
        let mut norm:T = T::zero();
        for x in 0..self.w {
            for y in 0..self.h {
                norm += self.get(0,0) * self.get(0,0);
            }
        }
        return norm;
    }
}
/*
struct ColumnPanelMatrix<T: Scalar> {
    h: usize,
    w: usize,
    panel_h: usize,
    panel_w: usize,
    panel_stride: usize,
    buffer: Vec<T>,
}
impl<T: Scalar> Matrix<T> for ColumnPanelMatrix<T> {
    fn get( &self, x: usize, y:usize ) -> T {
        let panel_id = y / self.panel_w;
        let panel_index  = y % self.panel_w;
        let elem_index = panel_id * self.panel_stride + x * self.panel_w + panel_index;

        return self.buffer[ elem_index ];
    }
    fn set( &mut self, x: usize, y:usize, alpha: T) -> () {
        let panel_id = y / self.panel_w;
        let panel_index  = y % self.panel_w;
        let elem_index = panel_id * self.panel_stride + x * self.panel_w + panel_index;

        self.buffer[ elem_index ] = alpha;
    }
    fn height( &self ) -> usize{
        return self.h;
    }
    fn width( &self ) -> usize{
        return self.w;
    }
}*/

struct View<T> {
    m: usize,
    n: usize,
    k: usize,
    m_off: usize,
    n_off: usize,
    k_off: usize,

    a_pack: Vec<T>,
    ap_orig_x: usize,
    ap_orig_y: usize,
    b_pack: Vec<T>,
    bp_orig_x: usize,
    bp_orig_y: usize,
    c_pack: Vec<T>,
    cp_orig_x: usize,
    cp_orig_y: usize,
}
impl<T:Scalar> View<T>{
    fn new( a: &Matrix<T>, b: &Matrix<T>, c: &Matrix<T> ) -> View<T>{
        let v : View<T> = View{ m: c.h, n: c.w, k: a.w,
                      m_off: 0, n_off: 0, k_off: 0,
                      a_pack: Vec::new(), ap_orig_x: 0, ap_orig_y: 0,
                      b_pack: Vec::new(), bp_orig_x: 0, bp_orig_y: 0,
                      c_pack: Vec::new(), cp_orig_x: 0, cp_orig_y: 0 };
        return v;
    }
}

trait GemmNode<T: Scalar> {
     fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>, v: &mut View<T> ) -> ();
}
/*
fn pack<T: Scalar>( a: &Matrix<T> ) {
}

struct PackA<T: Scalar, S: GemmNode<T>> {
    child: S,
    panel_height: usize,
    panel_width: usize,
    dt: PhantomData<T>,
}
impl<T:Scalar, S:GemmNode<T>> GemmNode<T> for PackA<T,S> {
     fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>, v: &mut View<T> ) {
        pack( a );
        self.child.run(a, b, c, v);
    }
}

struct PackB<T: Scalar, S: GemmNode<T>> {
    child: S,
    panel_height: usize,
    panel_width: usize,
    _dt: PhantomData<T>,
}
impl<T:Scalar, S:GemmNode<T>> GemmNode<T> for PackB<T,S> {
     fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>, v: &mut View<T> ) {
        pack( b );
        self.child.run(a, b, c, v);
    }
}
*/

struct PartM<T: Scalar, S: GemmNode<T>> {
    nb: usize,
    iota: usize,
    child: S,
    _dt: PhantomData<T>
}
impl<T: Scalar, S: GemmNode<T>> GemmNode<T> for PartM<T,S> {
    fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c:&mut Matrix<T>, v: &mut View<T> ) -> () {
        let m_save = v.m;
        let m_off_save = v.m_off;
        
        let mut i = 0;
        while i < m_save  {
            let nb_step = std::cmp::min( self.nb, m_save - i );   
            v.m = nb_step;
            v.m_off = m_off_save + i;
            self.child.run(a, b, c, v);
            i += nb_step;
        }
        v.m = m_save;
        v.m_off = m_off_save;
    }
}

//leaf node that does nothing for testing
struct End { }
impl<T: Scalar> GemmNode<T> for End {
    fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>, v: &mut View<T> ) -> () {
    }
}

struct TripleLoopKernel{}
impl<T: Scalar> GemmNode<T> for TripleLoopKernel {
    fn run( &self, a: &Matrix<T>, b: &Matrix<T>, c: &mut Matrix<T>, v: &mut View<T> ) -> () {
        //For now, let's do an axpy based gemm
        for jn in 0..v.n {
            for pk in 0..v.k {
                for im in 0..v.m {
                    let y = im+v.m_off;
                    let x = jn+v.n_off;
                    let z = pk+v.k_off;
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
    }
}

fn axpy<T:Scalar>( alpha: T, x: &Matrix<T>, beta: T, y: &mut Matrix<T> ) {
    //x.h should equal y.h and x.w should equal y.w!
    //but we dont check this yet
    for i in 0..x.w {
        for j in 0..x.h {
            let t = alpha * x.get(j,i) + beta * y.get(j,i);
            y.set(j,i,t); 
        }
    }
}

fn main() {
    let mut a : Matrix<f64> = Matrix::new(10, 10);
    let mut b : Matrix<f64> = Matrix::new(10, 10);
    let mut c : Matrix<f64> = Matrix::new(10, 10);
    let mut gemm_view : View<f64> = View::new(&a,&b,&c);
    a.fill_rand();
    b.fill_rand();
    c.fill_zero();


    let mut w : Matrix<f64> = Matrix::new(10, 1);
    let mut abw : Matrix<f64> = Matrix::new(10, 1);
    let mut bw : Matrix<f64> = Matrix::new(10, 1);
    let mut cw : Matrix<f64> = Matrix::new(10, 1);
    w.fill_rand();
    cw.fill_zero();
    bw.fill_zero();
    abw.fill_zero();

    
    let kernel : TripleLoopKernel = TripleLoopKernel{};
    //C = AB
    kernel.run( &a, &b, &mut c, &mut gemm_view );

    //Do bw = Bw, then abw = A*(Bw)   
    let mut bw_view : View<f64> = View::new(&b, &w, &bw);
    kernel.run( &b, &w, &mut bw, &mut bw_view );
    let mut abw_view : View<f64> = View::new(&a, &bw, &abw);
    kernel.run( &a, &bw, &mut abw, &mut abw_view );

    //Do cw = Cw
    let mut cw_view : View<f64> = View::new(&c, &w, &cw);
    kernel.run( &c, &w, &mut cw, &mut cw_view );
    
    //Cw -= abw
    axpy( 1.0, &abw, -1.0, &mut cw );
    let norm = cw.frosqr();
    println!("diff is {}", norm);
    /*

    let leaf : End = End{};
    let root : PartM<End> = PartM{child: leaf};
    root.run(&a, &b, &mut c);

    println!("c is {}", c.buffer);*/
}
