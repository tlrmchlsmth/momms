extern crate alloc;

use thread_comm::ThreadInfo;
use self::alloc::heap;
use matrix::{Scalar,Mat};

use core::{mem,cmp};
use core::ptr::{self};

#[derive(Copy,Clone)]
struct MatrixView {
    offset: usize,
    padding: usize,
    iter_size: usize,
}
impl MatrixView {
    #[inline(always)]
    fn physical_size( &self ) -> usize {
        if self.padding < self.iter_size {
            self.iter_size - self.padding
        } else {
            0   
        }
    }
}

pub struct Matrix<T: Scalar> {
    //Stack of views of the matrix
    y_views: Vec<MatrixView>,
    x_views: Vec<MatrixView>,
    
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
            let buf = heap::allocate( (h * w + 256) * mem::size_of::<T>(), 4096 );

            let mut y_views : Vec<MatrixView> = Vec::with_capacity( 16 );
            let mut x_views : Vec<MatrixView> = Vec::with_capacity( 16 );
            y_views.push(MatrixView{ offset: 0, padding: 0, iter_size: h });
            x_views.push(MatrixView{ offset: 0, padding: 0, iter_size: w });
        
            Matrix{ y_views: y_views,
                    x_views: x_views, 
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
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        self.buffer.offset((y_view.offset*self.row_stride + x_view.offset*self.column_stride) as isize)
    }

    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        self.buffer.offset((y_view.offset*self.row_stride + x_view.offset*self.column_stride) as isize)
    }
}
impl<T: Scalar> Mat<T> for Matrix<T> {
    #[inline(always)]
    fn get( &self, y: usize, x: usize) -> T {
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        let y_coord = (y + y_view.offset) * self.row_stride;
        let x_coord = (x + x_view.offset) * self.column_stride;
        unsafe{
            ptr::read( self.buffer.offset((y_coord + x_coord) as isize) )
        }
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x: usize, alpha: T) {
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        let y_coord = (y + y_view.offset) * self.row_stride;
        let x_coord = (x + x_view.offset) * self.column_stride;
        unsafe{
            ptr::write( self.buffer.offset((y_coord + x_coord) as isize), alpha );
        }
    }
    #[inline(always)]
    fn off_y( &self ) -> usize { 
        self.y_views.last().unwrap().offset 
    }
    #[inline(always)]
    fn off_x( &self ) -> usize {
        self.x_views.last().unwrap().offset 
    }

    //Still need these for parallel range but it should go away
    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) { 
        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.offset = off_y; 
    }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) {
        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.offset = off_x; 
    }

    #[inline(always)]
    fn iter_height( &self ) -> usize {
        self.y_views.last().unwrap().iter_size 
    }
    #[inline(always)]
    fn iter_width( &self ) -> usize { 
        self.x_views.last().unwrap().iter_size 
    }
    #[inline(always)]
    fn set_iter_height( &mut self, iter_h: usize ) {
        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.iter_size = iter_h; 
    }
    #[inline(always)]
    fn set_iter_width( &mut self, iter_w: usize ) {
        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.iter_size = iter_w; 
    }

    #[inline(always)]
    fn logical_h_padding( &self ) -> usize { 
        self.y_views.last().unwrap().padding 
    }
    #[inline(always)]
    fn logical_w_padding( &self ) -> usize { 
        self.x_views.last().unwrap().padding 
    }
    #[inline(always)]
    fn set_logical_h_padding( &mut self, h_pad: usize ) { 
        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.padding = h_pad 
    }
    #[inline(always)]
    fn set_logical_w_padding( &mut self, w_pad: usize ) {
        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.padding = w_pad 
    }

    #[inline(always)]
    fn push_x_view( &mut self, blksz: usize ) -> usize {
        let (zoomed_view, uz_iter_size) = { 
            let view = self.x_views.last().unwrap();
            let zoomed_iter_size = cmp::min(blksz, view.iter_size);

            //Need to figure out new padding
            let physical_size = view.physical_size();
            let zoomed_padding = if zoomed_iter_size < physical_size {
                0   
            } else {
                zoomed_iter_size - physical_size
            };  
            (MatrixView{ offset: view.offset, padding: zoomed_padding, iter_size: zoomed_iter_size }, view.iter_size)
        };  
        self.x_views.push(zoomed_view);
        uz_iter_size
    }
    #[inline(always)]
    fn push_y_view( &mut self, blksz: usize ) -> usize{
        let (zoomed_view, uz_iter_size) = { 
            let view = self.y_views.last().unwrap();
            let zoomed_iter_size = cmp::min(blksz, view.iter_size);

            //Need to figure out new padding
            let physical_size = view.physical_size();
            let zoomed_padding = if zoomed_iter_size < physical_size {
                0   
            } else {
                zoomed_iter_size - physical_size
            };  
            (MatrixView{ offset: view.offset, padding: zoomed_padding, iter_size: zoomed_iter_size }, view.iter_size)
        };  
        self.y_views.push(zoomed_view);
        uz_iter_size
    }
    #[inline(always)]
    fn pop_x_view( &mut self ) {
        debug_assert!( self.x_views.len() >= 2 );
        self.x_views.pop();
    }
    #[inline(always)]
    fn pop_y_view( &mut self ) {
        debug_assert!( self.y_views.len() >= 2 );
        self.y_views.pop();
    }
    #[inline(always)]
    fn slide_x_view_to( &mut self, x: usize, blksz: usize ) {
        let view_len = self.x_views.len();
        debug_assert!( view_len >= 2 );

        let unzoomed_view = self.x_views[view_len-2];
        let mut view = self.x_views.last_mut().unwrap();

        let new_iter_size = cmp::min(blksz, unzoomed_view.iter_size - x);
        let unzoomed_physical_size = unzoomed_view.physical_size();

        let new_padding = if x + new_iter_size < unzoomed_physical_size {
            0
        } else {
            x + new_iter_size - cmp::max(x, unzoomed_physical_size)
        };

        view.iter_size = new_iter_size;
        view.padding = new_padding;
        view.offset = unzoomed_view.offset + x;
    }
    #[inline(always)]
    fn slide_y_view_to( &mut self, y: usize, blksz: usize ) {
        let view_len = self.y_views.len();
        debug_assert!( view_len >= 2 );

        let unzoomed_view = self.y_views[view_len-2];
        let mut view = self.y_views.last_mut().unwrap();

        let new_iter_size = cmp::min(blksz, unzoomed_view.iter_size - y);
        let unzoomed_physical_size = unzoomed_view.physical_size();

        let new_padding = if y + new_iter_size < unzoomed_physical_size {
            0
        } else {
            y + new_iter_size - cmp::max(y, unzoomed_physical_size)
        };

        view.iter_size = new_iter_size;
        view.padding = new_padding;
        view.offset = unzoomed_view.offset + y;
    }

   
    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
       let x_view = self.x_views.last().unwrap();
       let y_view = self.y_views.last().unwrap();

       let mut x_views_alias : Vec<MatrixView> = Vec::with_capacity( 16 );
       let mut y_views_alias : Vec<MatrixView> = Vec::with_capacity( 16 );
       x_views_alias.push(MatrixView{ offset: x_view.offset, padding: x_view.offset, iter_size: x_view.iter_size });
       y_views_alias.push(MatrixView{ offset: y_view.offset, padding: y_view.offset, iter_size: y_view.iter_size });

        Matrix{ x_views: x_views_alias, y_views: y_views_alias,
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
