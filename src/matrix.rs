use std::ops::{Add, Mul, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::num::{Zero,One};
use core::fmt::{Display};
extern crate rand;

pub trait Scalar where
    Self: Add<Self, Output=Self>,
    Self: Mul<Self, Output=Self>,
    Self: Sub<Self, Output=Self>,
    Self: Div<Self, Output=Self>,
    Self: AddAssign<Self>,
    Self: MulAssign<Self>,
    Self: SubAssign<Self>,
    Self: DivAssign<Self>,
    Self: Zero,
    Self: One,
    Self: Sized,
    Self: Copy,
    Self: Display,
    Self: rand::Rand,
{}
impl Scalar for f64{}
impl Scalar for f32{}

/* Mat trait and its implementors */
pub trait Mat<T: Scalar> {
    #[inline(always)]
    fn get( &self, y: usize, x: usize) -> T;
    #[inline(always)]
    fn set( &mut self, y: usize, x: usize, alpha: T) -> ()
            where T: Sized;
    
    #[inline(always)]
    fn height( &self ) -> usize;
    #[inline(always)]
    fn width( &self ) -> usize;
    #[inline(always)]
    fn off_y( &self ) -> usize;
    #[inline(always)]
    fn off_x( &self ) -> usize;

    #[inline(always)]
    fn set_height( &mut self, h: usize ) -> ();
    #[inline(always)]
    fn set_width( &mut self, w: usize ) -> ();
    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) -> ();
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) -> ();

    fn print_wolfram( &self ) -> () {
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
            println!("");
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
        norm
    }
   
    fn copy_from( &mut self, other: &Mat<T>  ) {
        self.axpby( T::one(), other, T::zero() );
    }
    fn axpy( &mut self, alpha: T, other: &Mat<T> ) {
        self.axpby( alpha, other, T::one() );
    }
    
    fn axpby( &mut self, alpha: T, other: &Mat<T>, beta: T ) {
        if self.width() != other.width() || self.height() != other.height() {
            panic!("Cannot operate on nonconformal matrices!");
        }
        for x in 0..self.width() {
            for y in 0..self.height() { 
                let t = alpha * other.get(y,x) + beta * self.get(y,x);
                self.set(y,x,t);
            }
        }
    }
}

pub struct Matrix<T: Scalar> {
    //Height and width
    pub h: usize,
    pub w: usize,
    
    //This Matrix may be a submatrix within a larger one
    pub off_y: usize,
    pub off_x: usize,
    
    //Strides and buffer
    pub row_stride: usize,
    pub column_stride: usize,
    pub buffer: Vec<T>,
}
impl<T: Scalar> Matrix<T> {
    pub fn new( h: usize, w: usize ) -> Matrix<T> {
        let mut buf: Vec<T> = Vec::with_capacity( h * w );
        unsafe{ buf.set_len( h * w );}
        Matrix{ h: h, w: w, 
                off_y: 0, off_x: 0,
                row_stride: 1, column_stride: h, buffer: buf }
    }
    
    #[inline(always)]
    pub unsafe fn get_const_buffer( &self ) -> *const T { 
        self.buffer.as_ptr().offset((self.off_y*self.row_stride + self.off_x*self.column_stride) 
                                     as isize)
    }

    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        self.buffer.as_mut_ptr().offset((self.off_y*self.row_stride + self.off_x*self.column_stride) 
                                         as isize)
    }
}
impl<T: Scalar> Mat<T> for Matrix<T> {
    #[inline(always)]
    fn get( &self, y: usize, x: usize) -> T {
        self.buffer[(y+self.off_y)*self.row_stride + (x+self.off_x)*self.column_stride]
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x: usize, alpha: T) -> () {
        self.buffer[(y+self.off_y)*self.row_stride + (x+self.off_x)*self.column_stride] = alpha;
    }
    #[inline(always)]
    fn width( &self ) -> usize { self.w }
    #[inline(always)]
    fn height( &self ) -> usize { self.h }
    #[inline(always)]
    fn off_y( &self ) -> usize { self.off_y }
    #[inline(always)]
    fn off_x( &self ) -> usize { self.off_x }
    #[inline(always)]
    fn set_height( &mut self, h: usize ) { self.h = h; }
    #[inline(always)]
    fn set_width( &mut self, w: usize ) { self.w = w; }
    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) { self.off_y = off_y }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) { self.off_x = off_x }
}

pub struct ColumnPanelMatrix<T: Scalar> {
    pub h: usize,
    pub w: usize,
    
    pub off_y: usize,
    pub off_panel: usize,

    //Panel_h is always h
    pub panel_w: usize,
    pub panel_stride: usize,
    pub buffer:  Vec<T>,
}
impl<T: Scalar> ColumnPanelMatrix<T> {
    pub fn new( h: usize, w: usize, panel_w: usize ) -> ColumnPanelMatrix<T> {
        let mut n_panels = w / panel_w;
        if !(w % panel_w == 0) { 
            n_panels = w / panel_w + 1; 
        }

        let capacity = n_panels * panel_w * h;
        let mut buf: Vec<T> = Vec::with_capacity( capacity );
        unsafe{ buf.set_len( n_panels * panel_w * h );}
        ColumnPanelMatrix{ h: h, w: w,
                           off_y: 0, off_panel: 0,
                           panel_w: panel_w, panel_stride: panel_w*h, buffer: buf }
    }

    #[inline(always)]
    pub unsafe fn get_const_buffer( &self ) -> *const T { 
        self.buffer.as_ptr().offset((self.off_panel*self.panel_stride + self.off_y*self.panel_w) as isize)
    }
    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        self.buffer.as_mut_ptr().offset((self.off_panel*self.panel_stride + self.off_y*self.panel_w) as isize)
    }
}
impl<T: Scalar> Mat<T> for ColumnPanelMatrix<T> {
    #[inline(always)]
    fn get( &self, y: usize, x:usize ) -> T {
        let panel_id = x / self.panel_w;
        let panel_index  = x % self.panel_w;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (y + self.off_y) * self.panel_w + panel_index;
        self.buffer[ elem_index ]
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x:usize, alpha: T) -> () {
        let panel_id = x / self.panel_w;
        let panel_index  = x % self.panel_w;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (y + self.off_y) * self.panel_w + panel_index;

        self.buffer[ elem_index ] = alpha;
    }
    #[inline(always)]
    fn height( &self ) -> usize{ self.h }
    #[inline(always)]
    fn width( &self ) -> usize{ self.w }
    #[inline(always)]
    fn off_y( &self ) -> usize { self.off_y }
    #[inline(always)]
    fn off_x( &self ) -> usize { self.off_panel * self.panel_w }
    #[inline(always)]
    fn set_height( &mut self, h: usize ) { self.h = h; }
    #[inline(always)]
    fn set_width( &mut self, w: usize ) { self.w = w; }
    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) { self.off_y = off_y }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) { 
        if off_x % self.panel_w != 0 {
            println!("{} {}", off_x, self.panel_w);
            panic!("Illegal partitioning within ColumnPanelMatrix!");
        }
        self.off_panel = off_x / self.panel_w;
    }
}

pub struct RowPanelMatrix<T: Scalar> {
    pub h: usize,
    pub w: usize,

    pub off_x: usize,
    pub off_panel: usize,

    //Panel_h is always h
    pub panel_h: usize,
    pub panel_stride: usize,
    pub buffer: Vec<T>,
}
impl<T: Scalar> RowPanelMatrix<T> {
    pub fn new( h: usize, w: usize, panel_h: usize ) -> RowPanelMatrix<T> {
        let mut n_panels = h / panel_h;
        if !(h % panel_h == 0) { 
            n_panels = h / panel_h + 1; 
        }

        let capacity = n_panels * panel_h * w;
        
        let mut buf: Vec<T> = Vec::with_capacity( capacity );
        unsafe{ buf.set_len( n_panels * panel_h * w );}
        RowPanelMatrix{ h: h, w: w,
                        off_x: 0, off_panel: 0,
                        panel_h: panel_h, panel_stride: panel_h*w, buffer: buf }
    }
    #[inline(always)]
    pub unsafe fn get_const_buffer( &self ) -> *const T { 
        self.buffer.as_ptr().offset((self.off_panel*self.panel_stride + self.off_x*self.panel_h) as isize)
    }
    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        self.buffer.as_mut_ptr().offset((self.off_panel*self.panel_stride + self.off_x*self.panel_h) as isize)
    }
}
impl<T: Scalar> Mat<T> for RowPanelMatrix<T> {
    #[inline(always)]
    fn get( &self, y: usize, x:usize ) -> T {
        let panel_id = y / self.panel_h;
        let panel_index  = y % self.panel_h;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (x + self.off_x) * self.panel_h + panel_index;

        self.buffer[ elem_index ]
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x:usize, alpha: T) -> () {
        let panel_id = y / self.panel_h;
        let panel_index  = y % self.panel_h;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (x + self.off_x) * self.panel_h + panel_index;

        self.buffer[ elem_index ] = alpha;
    }
    #[inline(always)]
    fn height( &self ) -> usize{ self.h }
    #[inline(always)]
    fn width( &self ) -> usize{ self.w }
    #[inline(always)]
    fn off_y( &self ) -> usize { self.off_panel * self.panel_h }
    #[inline(always)]
    fn off_x( &self ) -> usize { self.off_x }
    #[inline(always)]
    fn set_height( &mut self, h: usize ) { self.h = h; }
    #[inline(always)]
    fn set_width( &mut self, w: usize ) { self.w = w; }
    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) { 
        if off_y % self.panel_h != 0 {
            println!("{} {}", off_y, self.panel_h);
            panic!("Illegal partitioning within RowPanelMatrix!");
        }
        self.off_panel = off_y / self.panel_h;
    }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) { self.off_x = off_x }
}
