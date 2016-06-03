#![feature(specialization)]
#![feature(zero_one)]
extern crate core;
extern crate rand;

use std::ops::{Add, Mul, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::num::{Zero};
use core::fmt::{Display};
use core::marker::{PhantomData, Sized};

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

/* Mat trait and its implementors */
trait Mat<T: Scalar> {
    fn get( &self, y: usize, x: usize) -> T;
    fn set( &mut self, y: usize, x: usize, alpha: T) -> ()
            where T: Sized;
    
    fn height( &self ) -> usize;
    fn width( &self ) -> usize;
    fn off_y( &self ) -> usize;
    fn off_x( &self ) -> usize;

    fn set_height( &mut self, h: usize ) -> ();
    fn set_width( &mut self, w: usize ) -> ();
    fn set_off_y( &mut self, off_y: usize ) -> ();
    fn set_off_x( &mut self, off_x: usize ) -> ();

    fn printWolfram( &self ) -> () {
        print!("{{");
        for y in 0..self.height() {
            print!("{{");
            for x in 0..self.width() {
                let formatted_number = format!("{:.*}", 2, self.get(y,x));
                print!("{}", formatted_number);
                if x < self.height() - 1 {
                    print!(",");
                }
            }
            print!("}}");
            if y < self.width() - 1 {
                print!(",");
            }
        }
        print!("}}");
    }

    fn print( &self ) -> () {
        for y in 0..self.width() {
            for x in 0..self.height() {
                let formatted_number = format!("{:.*}", 2, self.get(y,x));
                print!("{} ", formatted_number);
            }
        }
    }

    fn fill_rand( &mut self ) -> () {
        for x in 0..self.width() {
            for y in 0..self.height() {
                self.set(y,x,rand::random::<T>());
            }
        }
    }

    fn fill_zero( &mut self ) -> () {
        for x in 0..self.width() {
            for y in 0..self.height() {
                self.set(y,x,T::zero());
            }
        }
    }

    fn frosqr( &self ) -> T {
        let mut norm:T = T::zero();
        for x in 0..self.width() {
            for y in 0..self.height() {
                norm += self.get(y,x) * self.get(y,x);
            }
        }
        return norm;
    }
   
    fn copyFrom( &mut self, other: &Mat<T>  ) {
        if self.width() != other.width() || self.height() != other.height() {
            panic!("Cannot copy nonconformal matrices!");
        }
        for x in 0..self.width() {
            for y in 0..self.height() { 
                self.set(y,x,other.get(y,x));
            }
        }
    }
                    
}

struct Matrix<T: Scalar> {
    //Height and width
    h: usize,
    w: usize,
    
    //This Matrix may be a submatrix within a larger one
    off_y: usize,
    off_x: usize,
    
    //Strides and buffer
    row_stride: usize,
    column_stride: usize,
    buffer: Vec<T>,
}
impl<T: Scalar> Matrix<T> {
    fn new( h: usize, w: usize ) -> Matrix<T> {
        let mut buf = Vec::with_capacity( h * w );
        unsafe{ buf.set_len( h * w );}
        let mat = Matrix{ h: h, w: w, 
                off_y: 0, off_x: 0,
                row_stride: 1, column_stride: h, buffer: buf };
        return mat;
    }
}
impl<T: Scalar> Mat<T> for Matrix<T> {
    fn get( &self, y: usize, x: usize) -> T {
        return self.buffer[(y+self.off_y)*self.row_stride + (x+self.off_x)*self.column_stride];
    }
    fn set( &mut self, y: usize, x: usize, alpha: T) -> () {
        self.buffer[(y+self.off_y)*self.row_stride + (x+self.off_x)*self.column_stride] = alpha;
    }
    fn width( &self ) -> usize { return self.w; }
    fn height( &self ) -> usize { return self.h; }
    fn off_y( &self ) -> usize { return self.off_y; }
    fn off_x( &self ) -> usize { return self.off_x; }
    fn set_height( &mut self, h: usize ) { self.h = h; }
    fn set_width( &mut self, w: usize ) { self.w = w; }
    fn set_off_y( &mut self, off_y: usize ) { self.off_y = off_y }
    fn set_off_x( &mut self, off_x: usize ) { self.off_x = off_x }
}

struct ColumnPanelMatrix<T: Scalar> {
    h: usize,
    w: usize,

    off_y: usize,
    off_panel: usize,

    //Panel_h is always h
    panel_w: usize,
    panel_stride: usize,
    buffer: Vec<T>,
}
impl<T: Scalar> ColumnPanelMatrix<T> {
    fn new( h: usize, w: usize, panel_w: usize ) -> ColumnPanelMatrix<T> {
        let n_panels = w / panel_w;
        if !(w % panel_w) == 0 { 
            let n_panels = w / panel_w + 1; 
        }

        let capacity = n_panels * panel_w * h;
        
        let mut buf = Vec::with_capacity( capacity );
        unsafe{ buf.set_len( h * w );}
        let mat = ColumnPanelMatrix{ h: h, w: w,
                off_y: 0, off_panel: 0,
                panel_w: panel_w, panel_stride: panel_w*h, buffer: buf };
        return mat;
    }
}
impl<T: Scalar> Mat<T> for ColumnPanelMatrix<T> {
    fn get( &self, y: usize, x:usize ) -> T {
        let panel_id = x / self.panel_w;
        let panel_index  = x % self.panel_w;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (y + self.off_y) * self.panel_w + panel_index;

        return self.buffer[ elem_index ];
    }
    fn set( &mut self, y: usize, x:usize, alpha: T) -> () {
        let panel_id = y / self.panel_w;
        let panel_index  = y % self.panel_w;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (y + self.off_y) * self.panel_w + panel_index;

        self.buffer[ elem_index ] = alpha;
    }
    fn height( &self ) -> usize{ return self.h; }
    fn width( &self ) -> usize{ return self.w; }
    fn off_y( &self ) -> usize { return self.off_y; }
    fn off_x( &self ) -> usize {
        return self.off_panel * self.panel_w; 
    }
    fn set_height( &mut self, h: usize ) { self.h = h; }
    fn set_width( &mut self, w: usize ) { self.w = w; }
    fn set_off_y( &mut self, off_y: usize ) { self.off_y = off_y }
    fn set_off_x( &mut self, off_x: usize ) { 
        if off_x % self.panel_w != 0 {
            println!("{} {}", off_x, self.panel_w);
            panic!("Illegal partitioning within ColumnPanelMatrix!");
        }
        self.off_panel = off_x / self.panel_w;
    }
}

trait GemmNode<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> {
     fn run( &self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> ();
}

struct PackACP<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    child: S,
    panel_width: usize,
    _dt: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    PackACP <T,At,Bt,Ct,S> {
    fn new( panel_width: usize, child: S ) -> PackACP<T, At, Bt, Ct, S>{
        return PackACP{ panel_width: panel_width, child: child, 
            _dt: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }; 
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackACP<T, At, Bt, Ct, S> {
    fn run( &self, a: &mut At, b: &mut Bt, c:&mut Ct ) -> () {
        let mut a_pack : ColumnPanelMatrix<T> = ColumnPanelMatrix::new( a.height(), a.width(), self.panel_width );

        //pack( a );
        self.child.run(a, b, c);
    }
}

struct PartM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    iota: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartM<T,At,Bt,Ct,S> {
    fn new( bsz: usize, iota: usize, child: S ) -> PartM<T, At, Bt, Ct, S>{
        return PartM{ bsz: bsz, iota: iota, child: child, 
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }; 
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartM<T,At,Bt,Ct,S> {
    fn run( &self, a: &mut At, b: &mut Bt, c:&mut Ct ) -> () {
        let m_save = c.height();
        let ay_off_save = a.off_y();
        let cy_off_save = c.off_y();
        
        let mut i = 0;
        while i < m_save  {
            let bsz_step = std::cmp::min( self.bsz, m_save - i );
            a.set_height( bsz_step );
            a.set_off_y( ay_off_save + i );
            c.set_height( bsz_step );
            c.set_off_y( cy_off_save + i );

            self.child.run(a, b, c);
            i += bsz_step;
        }
        a.set_height( m_save );
        a.set_off_y( ay_off_save );
        c.set_height( m_save );
        c.set_off_y( cy_off_save );
    }
}

struct PartN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    iota: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartN<T,At,Bt,Ct,S> {
    fn new( bsz: usize, iota: usize, child: S ) -> PartN<T, At, Bt, Ct, S>{
        return PartN{ bsz: bsz, iota: iota, child: child, 
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }; 
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartN<T,At,Bt,Ct,S> {
    fn run( &self, a: &mut At, b: &mut Bt, c:&mut Ct ) -> () {
        let n_save = c.width();
        let bx_off_save = b.off_x();
        let cx_off_save = c.off_x();
        
        let mut i = 0;
        while i < n_save  {
            let bsz_step = std::cmp::min( self.bsz, n_save - i );
            b.set_width( bsz_step );
            b.set_off_x( bx_off_save + i );
            c.set_width( bsz_step );
            c.set_off_x( cx_off_save + i );

            self.child.run(a, b, c);
            i += bsz_step;
        }
        b.set_width( n_save );
        b.set_off_x( bx_off_save );
        c.set_width( n_save );
        c.set_off_x( cx_off_save );
    }
}

struct PartK<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    bsz: usize,
    iota: usize,
    child: S,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> PartK<T,At,Bt,Ct,S> {
    fn new( bsz: usize, iota: usize, child: S ) -> PartK<T, At, Bt, Ct, S>{
        return PartK{ bsz: bsz, iota: iota, child: child, 
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }; 
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PartK<T,At,Bt,Ct,S> {
    fn run( &self, a: &mut At, b: &mut Bt, c:&mut Ct ) -> () {
        let k_save = a.width();
        let ax_off_save = a.off_x();
        let by_off_save = b.off_y();
        
        let mut i = 0;
        while i < k_save  {
            let bsz_step = std::cmp::min( self.bsz, k_save - i );
            a.set_width( bsz_step );
            a.set_off_x( ax_off_save + i );
            b.set_height( bsz_step );
            b.set_off_y( by_off_save + i );

            self.child.run(a, b, c);
            i += bsz_step;
        }
        a.set_width( k_save );
        a.set_off_x( ax_off_save );
        b.set_height( k_save );
        b.set_off_y( by_off_save );
    }
}

struct TripleLoopKernel{}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for TripleLoopKernel {
    fn run( &self, a: &mut At, b: &mut Bt, c: &mut Ct ) -> () {
        //For now, let's do an axpy based gemm
        for x in 0..c.width() {
            for z in 0..a.width() {
                for y in 0..c.height() {
/*                    let y = im+v.m_off;
                    let x = jn+v.n_off;
                    let z = pk+v.k_off;*/
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }
            }
        }
    }
}

fn axpy<T:Scalar, Xt: Mat<T>, Yt: Mat<T>>( alpha: T, x: &Xt, beta: T, y: &mut Yt ) {
    //x.h should equal y.h and x.w should equal y.w!
    //but we dont check this yet
    for i in 0..x.width() {
        for j in 0..x.height() {
            let t = alpha * x.get(j,i) + beta * y.get(j,i);
            y.set(j,i,t); 
        }
    }
}

fn main() {
    let m = 10;
    let n = 10;
    let k = 10;

    let mut a : ColumnPanelMatrix<f64> = ColumnPanelMatrix::new(m, k, 2);
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
    let loop1 : PartM<f64, ColumnPanelMatrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel> = PartM::new(5, 1, kernel);
    let loop2 : PartK<f64, ColumnPanelMatrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, ColumnPanelMatrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>> = PartK::new(4, 1, loop1);
    let loop3 : PartN<f64, ColumnPanelMatrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartK<f64, ColumnPanelMatrix<f64>, Matrix<f64>, Matrix<f64>, 
                PartM<f64, ColumnPanelMatrix<f64>, Matrix<f64>, Matrix<f64>, TripleLoopKernel>>> = PartN::new(5, 1, loop2);

    //If you instead delcare the phantom data here, the type inference system works
    //I.E. the compiler can infer the type of loop1_inf when you use that type in loop2_inf
    //Now the code grows linearly with the number of steps in our control tree
    //The problem with this is that we have all this ugly phantom data laying around!
    let kernel_inf = TripleLoopKernel{};
    let loop1_inf = PartM{ bsz: 5, iota: 1, child: kernel_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop2_inf = PartK{ bsz: 4, iota: 1, child: loop1_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };
    let loop3_inf = PartN{ bsz: 5, iota: 1, child: loop2_inf, 
        _t: PhantomData::<f64>, _at: PhantomData::<ColumnPanelMatrix<f64>>, _bt: PhantomData::<Matrix<f64>>, _ct: PhantomData::<Matrix<f64>> };

    let ref_gemm : TripleLoopKernel = TripleLoopKernel{};
    //C = AB
    loop3_inf.run( &mut a, &mut b, &mut c );
    //loop3.run( &mut a, &mut b, &mut c );

    a.printWolfram();
    println!("");
    b.printWolfram();
    println!("");
    c.printWolfram();
    println!("");

    //Do bw = Bw, then abw = A*(Bw)   
    ref_gemm.run( &mut b, &mut w, &mut bw );
    ref_gemm.run( &mut a, &mut bw, &mut abw );

    //Do cw = Cw
    ref_gemm.run( &mut c, &mut w, &mut cw );
    
    //Cw -= abw
    axpy( 1.0, &abw, -1.0, &mut cw );
    let norm = cw.frosqr();
    println!("diff is {}", norm);
}
