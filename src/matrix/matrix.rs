extern crate rand;

use core::fmt::Display;
use core::ops::{Add, Mul, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use thread_comm::ThreadInfo;

//Trait Definitions
pub trait ScalarConstants {
    #[inline(always)]
    fn one( ) -> Self;
    #[inline(always)]
    fn zero( ) -> Self;
}
pub trait Scalar where
    Self: Add<Self, Output=Self>,
    Self: Mul<Self, Output=Self>,
    Self: Sub<Self, Output=Self>,
    Self: Div<Self, Output=Self>,
    Self: AddAssign<Self>,
    Self: MulAssign<Self>,
    Self: SubAssign<Self>,
    Self: DivAssign<Self>,
    Self: Sized,
    Self: Copy,
    Self: Display,
    Self: rand::Rand,
    Self: Send,
    Self: ScalarConstants,
{}


impl ScalarConstants for f64 {
    #[inline(always)]
    fn one( ) -> Self {
        1.0 as f64
    }
    #[inline(always)]
    fn zero( ) -> Self {
        0.0 as f64
    }
}
impl Scalar for f64{}

impl ScalarConstants for f32 {
    #[inline(always)]
    fn one( ) -> Self {
        1.0 as f32
    }
    #[inline(always)]
    fn zero( ) -> Self {
        0.0 as f32
    }
}
impl Scalar for f32{}

/* Mat trait and its implementors */
pub trait Mat<T: Scalar> where Self: Send {
    #[inline(always)]
    fn get( &self, y: usize, x: usize) -> T;
    #[inline(always)]
    fn set( &mut self, y: usize, x: usize, alpha: T) 
            where T: Sized;

    #[inline(always)]
    fn height( &self ) -> usize {
        if self.iter_height() < self.logical_h_padding() {
            0
        } else {
            self.iter_height() - self.logical_h_padding()
        }
    }
    #[inline(always)]
    fn width( &self ) -> usize {
        if self.iter_width() < self.logical_w_padding() {
            0
        } else {
            self.iter_width() - self.logical_w_padding()
        }
    }

    #[inline(always)]
    fn off_y( &self ) -> usize;
    #[inline(always)]
    fn off_x( &self ) -> usize;

    #[inline(always)]
    fn iter_height( &self ) -> usize;
    #[inline(always)]
    fn iter_width( &self ) -> usize;

    #[inline(always)]
    fn set_iter_height( &mut self, h: usize );
    #[inline(always)]
    fn set_iter_width( &mut self, w: usize );

    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize );
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize );

    #[inline(always)]
    fn set_logical_h_padding( &mut self, iter_h: usize );
    #[inline(always)]
    fn logical_h_padding( &self ) -> usize;
    #[inline(always)]
    fn set_logical_w_padding( &mut self, iter_h: usize );
    #[inline(always)]
    fn logical_w_padding( &self ) -> usize;
    
    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self where Self: Sized;
    #[inline(always)]
    unsafe fn send_alias( &mut self, thr: &ThreadInfo<T> ); 

    fn push_x_view( &mut self, blksz: usize ) -> usize;
    fn push_y_view( &mut self, blksz: usize ) -> usize;
    fn pop_x_view( &mut self );
    fn pop_y_view( &mut self );
    fn slide_x_view_to( &mut self, x: usize, blksz: usize );
    fn slide_y_view_to( &mut self, y: usize, blksz: usize );

    fn print_wolfram( &self ) {
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

    fn print_matlab( &self ) {
        print!("[");
        for y in 0..self.height() {
            for x in 0..self.width() {
                print!("{}", self.get(y,x));
                if x+1 != self.width() { print!(","); }
            }
            if y+1 != self.height() { print!(";"); }
        }
        print!("]");
    }

    fn print( &self ) {
        for y in 0..self.height() {
            for x in 0..self.width() {
                let formatted_number = format!("{:.*}", 3, self.get(y,x));
                print!("{} ", formatted_number);
            }
            println!("");
        }
    }

    fn fill_rand( &mut self ) {
        for x in 0..self.width() {
            for y in 0..self.height() {
                self.set(y,x,rand::random::<T>());
            }
        }
    }

    fn fill_zero( &mut self ) {
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

    fn axpby_base( &mut self, alpha: T, other: &Mat<T>, beta: T, 
                  off_y: usize, off_x: usize, h: usize, w: usize ) {
        for x in off_x..off_x+w {
            for y in off_y..off_y+h { 
                let t = alpha * other.get(y,x) + beta * self.get(y,x);
                self.set(y,x,t);
            }
        }
    }

    //Split into quarters recursivly for cache oblivious axpby
    fn axpby_rec( &mut self, alpha: T, other: &Mat<T>, beta: T, 
                  off_y: usize, off_x: usize, h: usize, w: usize ) {
        if h < 32 || w < 32 { self.axpby_base( alpha, other, beta, off_y, off_x, h, w); }
        else{
            let half_h = h / 2;
            let half_w = w / 2;
            self.axpby_rec(alpha, other, beta, off_y, off_x, half_h, half_w );
            self.axpby_rec(alpha, other, beta, off_y + half_h, off_x, h - half_h, half_w );
            self.axpby_rec(alpha, other, beta, off_y, off_x + half_w, half_h, w - half_w );
            self.axpby_rec(alpha, other, beta, off_y + half_h, off_x + half_w, h - half_h, w - half_w );
        }
    }
    
    fn axpby( &mut self, alpha: T, other: &Mat<T>, beta: T ) {
        if self.width() != other.width() || self.height() != other.height() {
            panic!("Cannot operate on nonconformal matrices!");
        }
        let h = self.height();
        let w = self.width();
        self.axpby_rec(alpha, other, beta, 0, 0, h, w); 
    }

    fn axpby_small( &mut self, alpha: T, other: &Mat<T>, beta: T ) {
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

pub trait ResizableBuffer<T: Scalar> {
    fn empty() -> Self;
    fn capacity(&self) -> usize;
    fn set_capacity(&mut self, capacity: usize); 
    fn capacity_for(other: &Mat<T>) -> usize;
    fn aquire_buffer_for(&mut self, capacity: usize );
    fn resize_to( &mut self, other: &Mat<T> ); 
}
