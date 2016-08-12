extern crate alloc;

use thread::ThreadInfo;
use self::alloc::heap;
use matrix::{Scalar,Mat};

use core::mem;
use core::ptr::{self};

pub struct Matrix<T: Scalar> {
    //Height and width for iteration space
    iter_h: usize,
    iter_w: usize,

    //Padding to iteration h and w
    h_padding: usize,
    w_padding: usize,
    
    //This Matrix may be a submatrix within a larger one
    off_y: usize,
    off_x: usize,
    
    //Strides and buffer
    row_stride: usize,
    column_stride: usize,
    buffer: *mut T,
    capacity: usize,
    is_alias: bool,
}
impl<T: Scalar> Matrix<T> {
    pub fn new( h: usize, w: usize ) -> Matrix<T> {
        assert!(mem::size_of::<T>() != 0, "Matrix can't handle ZSTs");
        unsafe { 
            let buf = heap::allocate( h * w * mem::size_of::<T>(), 4096 );
        
            Matrix{ iter_h: h, iter_w: w, 
                    h_padding: 0, w_padding: 0,
                    off_y: 0, off_x: 0,
                    row_stride: 1, column_stride: h,
                    buffer: buf as *mut _,
                    capacity: h * w,
                    is_alias: false }
        }
    }

    #[inline(always)] pub fn get_row_stride( &self ) -> usize { self.row_stride }
    #[inline(always)] pub fn get_column_stride( &self ) -> usize { self.column_stride }

    
    #[inline(always)]
    pub unsafe fn get_buffer( &self ) -> *const T { 
        self.buffer.offset((self.off_y*self.row_stride + self.off_x*self.column_stride) as isize)
    }

    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        self.buffer.offset((self.off_y*self.row_stride + self.off_x*self.column_stride) as isize)
    }
}
impl<T: Scalar> Mat<T> for Matrix<T> {
    #[inline(always)]
    fn get( &self, y: usize, x: usize) -> T {
        let y_coord = (y + self.off_y) * self.row_stride;
        let x_coord = (x + self.off_x) * self.column_stride;
        unsafe{
            ptr::read( self.buffer.offset((y_coord + x_coord) as isize) )
        }
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x: usize, alpha: T) {
        let y_coord = (y + self.off_y) * self.row_stride;
        let x_coord = (x + self.off_x) * self.column_stride;
        unsafe{
            ptr::write( self.buffer.offset((y_coord + x_coord) as isize), alpha );
        }
    }
    #[inline(always)]
    fn off_y( &self ) -> usize { self.off_y }
    #[inline(always)]
    fn off_x( &self ) -> usize { self.off_x }
/*    #[inline(always)]
    fn set_height( &mut self, h: usize ) { self.h = h; }
    #[inline(always)]
    fn set_width( &mut self, w: usize ) { self.w = w; }*/
    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) { self.off_y = off_y }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) { self.off_x = off_x }

    #[inline(always)]
    fn iter_height( &self ) -> usize { self.iter_h }
    #[inline(always)]
    fn iter_width( &self ) -> usize { self.iter_w }
    #[inline(always)]
    fn set_iter_height( &mut self, iter_h: usize ) { self.iter_h = iter_h; }
    #[inline(always)]
    fn set_iter_width( &mut self, iter_w: usize ) { self.iter_w = iter_w; }

    #[inline(always)]
    fn set_logical_h_padding( &mut self, h_pad: usize ) { self.h_padding = h_pad }
    #[inline(always)]
    fn get_logical_h_padding( &self ) -> usize { self.h_padding }
    #[inline(always)]
    fn set_logical_w_padding( &mut self, w_pad: usize ) { self.w_padding = w_pad }
    #[inline(always)]
    fn get_logical_w_padding( &self ) -> usize { self.w_padding }
    
    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
        Matrix{ iter_h: self.iter_h, iter_w: self.iter_w, 
                h_padding: self.h_padding, w_padding: self.w_padding,
                off_y: self.off_y, off_x: self.off_x,
                row_stride: self.row_stride, column_stride: self.column_stride,
                buffer: self.buffer,
                capacity: self.capacity,
                is_alias: true }
    }

    #[inline(always)]
    unsafe fn send_alias( &mut self, thr: &ThreadInfo<T> ) {
        let buf = thr.broadcast( self.buffer );
        self.is_alias = true;
        self.buffer = buf;
    }
}
impl<T:Scalar> Drop for Matrix<T> {
    fn drop(&mut self) {
        unsafe {
            if !self.is_alias {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
            }
        }
    }
}
unsafe impl<T:Scalar> Send for Matrix<T> {}
