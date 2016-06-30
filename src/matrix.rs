extern crate rand;
extern crate alloc;

use core::ptr::{self};
use core::mem;
use self::alloc::heap;

use core::ops::{Add, Mul, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use core::num::{Zero,One};
use core::fmt::{Display};
use core::cmp;
use core::marker::{PhantomData};

use thread::ThreadInfo;

use typenum::{Unsigned};

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
    Self: Send,
{}
impl Scalar for f64{}
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
        if self.iter_height() < self.get_logical_h_padding() {
            0
        } else {
            self.iter_height() - self.get_logical_h_padding()
        }
    }
    #[inline(always)]
    fn width( &self ) -> usize {
        if self.iter_width() < self.get_logical_w_padding() {
            0
        } else {
            self.iter_width() - self.get_logical_w_padding()
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
    fn get_logical_h_padding( &self ) -> usize;
    #[inline(always)]
    fn set_logical_w_padding( &mut self, iter_h: usize );
    #[inline(always)]
    fn get_logical_w_padding( &self ) -> usize;
    
    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self where Self: Sized;
    #[inline(always)]
    unsafe fn send_alias( &self, thr: &ThreadInfo<T> ) -> Self where Self: Sized;
    
    #[inline(always)]
    fn adjust_y_view( &mut self, parent_iter_h: usize, parent_off_y: usize,
                      target_h: usize, index: usize ) {
        debug_assert!( index < parent_iter_h, "Error adjusting y view!" );
        let new_iter_h = cmp::min( target_h, parent_iter_h - index );
        self.set_iter_height( new_iter_h );
        self.set_off_y( parent_off_y + index );
    }

    #[inline(always)]
    fn adjust_x_view( &mut self, parent_iter_w: usize, parent_off_x: usize,
                      target_w: usize, index: usize ) {
        debug_assert!( index < parent_iter_w, "Error adjusting y view!" );
        let new_iter_w = cmp::min( target_w, parent_iter_w - index );
        self.set_iter_width( new_iter_w );
        self.set_off_x( parent_off_x + index );
    }

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

    fn print( &self ) {
        for y in 0..self.height() {
            for x in 0..self.width() {
                let formatted_number = format!("{:.*}", 2, self.get(y,x));
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
/*        for x in 0..self.width() {
            for y in 0..self.height() { 
                let t = alpha * other.get(y,x) + beta * self.get(y,x);
                self.set(y,x,t);
            }
        }
*/
        let h = self.height();
        let w = self.width();
        self.axpby_rec(alpha, other, beta, 0, 0, h, w); 
    }
}

pub trait ResizeableBuffer<T: Scalar> {
    fn empty() -> Self;
    fn resize_to_fit(&mut self, other: &Mat<T>);
}

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
    unsafe fn send_alias( &self, thr: &ThreadInfo<T> ) -> Self {
        let buf = thr.broadcast( self.buffer );

        Matrix{ iter_h: self.iter_h, iter_w: self.iter_w, 
                h_padding: self.h_padding, w_padding: self.w_padding,
                off_y: self.off_y, off_x: self.off_x,
                row_stride: self.row_stride, column_stride: self.column_stride,
                buffer: buf as *mut _,
                capacity: self.capacity,
                is_alias: true }
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

pub struct ColumnPanelMatrix<T: Scalar, PW: Unsigned> {
    //Height and width for iteration space
    iter_h: usize,
    iter_w: usize,

    //Height and width padding for equal iteration spaces
    logical_h_padding: usize,
    logical_w_padding: usize,
    
    off_y: usize,
    off_panel: usize,

    //Panel_h is always h
//    panel_w: usize,

    n_panels: usize,    //Physical number of panels
    panel_stride: usize,
    
    buffer: *mut T,
    capacity: usize,

    _pwt: PhantomData<PW>,
}
impl<T: Scalar, PW: Unsigned> ColumnPanelMatrix<T,PW> {
    pub fn new( h: usize, w: usize ) -> ColumnPanelMatrix<T,PW> {
        assert!(mem::size_of::<T>() != 0, "Matrix can't handle ZSTs");
        let panel_w = PW::to_usize();

        let mut n_panels = w / panel_w; 
        if !(w % panel_w == 0) { 
            n_panels = w / panel_w + 1; 
        }
        let capacity = n_panels * panel_w * h;
        unsafe { 
            let ptr = heap::allocate( capacity * mem::size_of::<T>(), 4096 );

            ColumnPanelMatrix{ iter_h: h, iter_w: w,
                           logical_h_padding: 0, logical_w_padding: 0,
                           off_y: 0, off_panel: 0,
                           n_panels: n_panels, panel_stride: panel_w*h, 
                           buffer: ptr as *mut _, capacity: capacity,
                           _pwt: PhantomData }
        }
    }
    

    #[inline(always)]
    pub fn get_n_panels( &self ) -> usize { self.n_panels }

    #[inline(always)]
    pub fn get_panel_stride( &self ) -> usize { self.panel_stride }
    
    #[inline(always)]
    pub fn get_panel_w( &self ) -> usize { PW::to_usize() }

    #[inline(always)]
    pub unsafe fn get_buffer( &self ) -> *const T { 
        let panel_w = PW::to_usize();
        self.buffer.offset((self.off_panel*self.panel_stride + self.off_y*panel_w) as isize)
    }

    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        let panel_w = PW::to_usize();
        self.buffer.offset((self.off_panel*self.panel_stride + self.off_y*panel_w) as isize)
    }

    #[inline(always)]
    pub unsafe fn get_panel( &mut self, id: usize ) -> *mut T {
        self.buffer.offset(((self.off_panel + id)*self.panel_stride) as isize)
    }
}
impl<T: Scalar, PW: Unsigned> Mat<T> for ColumnPanelMatrix<T, PW> {
    #[inline(always)]
    fn get( &self, y: usize, x: usize ) -> T {
        let panel_w = PW::to_usize();
        let panel_id = x / panel_w;
        let panel_index  = x % panel_w;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (y + self.off_y) * panel_w + panel_index;
        unsafe{
            ptr::read( self.buffer.offset(elem_index as isize ) )
        }
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x: usize, alpha: T) {
        let panel_w = PW::to_usize();
        let panel_id = x / panel_w;
        let panel_index  = x % panel_w;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (y + self.off_y) * panel_w + panel_index;
        unsafe{
            ptr::write( self.buffer.offset(elem_index as isize ), alpha );
        }
    }
    
    #[inline(always)]
    fn off_y( &self ) -> usize { self.off_y }
    #[inline(always)]
    fn off_x( &self ) -> usize { 
        let panel_w = PW::to_usize();
        self.off_panel * panel_w 
    }

    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) { self.off_y = off_y }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) { 
        let panel_w = PW::to_usize();
        if off_x % panel_w != 0 {
            println!("{} {}", off_x, panel_w);
            panic!("Illegal partitioning within ColumnPanelMatrix!");
        }
        self.off_panel = off_x / panel_w;
    }

    #[inline(always)]
    fn iter_height( &self ) -> usize { self.iter_h }
    #[inline(always)]
    fn iter_width( &self ) -> usize { self.iter_w }
    #[inline(always)]
    fn set_iter_height( &mut self, iter_h: usize ) { self.iter_h = iter_h; }
    #[inline(always)]
    fn set_iter_width( &mut self, iter_w: usize ) { self.iter_w = iter_w; }

    #[inline(always)]
    fn set_logical_h_padding( &mut self, h_pad: usize ) { self.logical_h_padding = h_pad }
    #[inline(always)]
    fn get_logical_h_padding( &self ) -> usize { self.logical_h_padding }
    #[inline(always)]
    fn set_logical_w_padding( &mut self, w_pad: usize ) { self.logical_w_padding = w_pad }
    #[inline(always)]
    fn get_logical_w_padding( &self ) -> usize { self.logical_w_padding }
    
    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
        ColumnPanelMatrix{ iter_h: self.iter_h, iter_w: self.iter_w,
                           logical_h_padding: self.logical_h_padding, 
                           logical_w_padding: self.logical_w_padding,
                           off_y: self.off_y, off_panel: self.off_panel,
                           n_panels: self.n_panels,
                           panel_stride: self.panel_stride,
                           buffer: self.buffer, 
                           capacity: self.capacity,
                           _pwt: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias( &self, thr: &ThreadInfo<T> ) -> Self {
        let buf = thr.broadcast( self.buffer );

        ColumnPanelMatrix{ iter_h: self.iter_h, iter_w: self.iter_w,
                           logical_h_padding: self.logical_h_padding, 
                           logical_w_padding: self.logical_w_padding,
                           off_y: self.off_y, off_panel: self.off_panel,
                           n_panels: self.n_panels,
                           panel_stride: self.panel_stride,
                           buffer: buf, 
                           capacity: self.capacity,
                           _pwt: PhantomData }
    }
}
impl<T:Scalar, PW: Unsigned> Drop for ColumnPanelMatrix<T, PW> {
    fn drop(&mut self) {
        unsafe {
            heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
        }
    }
}

unsafe impl<T:Scalar, PW: Unsigned> Send for ColumnPanelMatrix<T, PW> {}

impl<T:Scalar, PW: Unsigned> ResizeableBuffer<T> for ColumnPanelMatrix<T, PW> {
    #[inline(always)]
    fn empty() -> Self {
        ColumnPanelMatrix::new(0,0)
    }
    #[inline(always)]
    fn resize_to_fit( &mut self, other: &Mat<T> ) {
        let panel_w = PW::to_usize();

        if self.off_y != 0 || self.off_panel != 0 { panic!("can't resize a submatrix!"); }

        let mut new_n_panels = other.width() / panel_w;
        if other.width() % panel_w != 0 { new_n_panels += 1; }

        let mut req_capacity = new_n_panels * panel_w * other.height();
        let old_capacity = self.capacity;
        if req_capacity > old_capacity {
            unsafe {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * old_capacity, 4096);
                let newbuf = heap::allocate( req_capacity * mem::size_of::<T>(), 4096 );
                self.buffer = newbuf as *mut _;
                self.capacity = req_capacity;
            }
        }

        self.iter_h = other.iter_height();
        self.iter_w = other.iter_width();
        self.logical_h_padding = other.get_logical_h_padding();
        self.logical_w_padding = other.get_logical_w_padding();

        self.n_panels = new_n_panels;
        self.panel_stride = panel_w*other.height();
    }
    
}

pub struct RowPanelMatrix<T: Scalar, PH: Unsigned> {
    //Height and width padding for equal iteration spaces
    iter_h: usize,
    iter_w: usize,

    //Height and width padding for equal iteration spaces
    logical_h_padding: usize,
    logical_w_padding: usize,

    off_x: usize,
    off_panel: usize,

    //Panel_w is always w
    //panel_h: usize,
    n_panels: usize,
    panel_stride: usize,

    buffer: *mut T,
    capacity: usize,

    _pht: PhantomData<PH>,
}
impl<T: Scalar, PH: Unsigned> RowPanelMatrix<T, PH> {
    pub fn new( h: usize, w: usize ) -> RowPanelMatrix<T, PH> {
        assert!(mem::size_of::<T>() != 0, "Matrix can't handle ZSTs");
        let panel_h = PH::to_usize();

        let mut n_panels = h / panel_h;
        if !(h % panel_h == 0) { 
            n_panels = h / panel_h + 1; 
        }
        
        let capacity = n_panels * panel_h * w;
        unsafe { 
            let ptr = heap::allocate( capacity * mem::size_of::<T>(), 4096 );

            RowPanelMatrix{ iter_h: h, iter_w: w,
                            logical_h_padding: 0, logical_w_padding: 0,
                            off_x: 0, off_panel: 0,
                            n_panels : n_panels, panel_stride: panel_h*w, 
                            buffer: ptr as *mut _, capacity: capacity,
                            _pht: PhantomData }
        }
    }


    #[inline(always)]
    pub fn get_n_panels( &self ) -> usize { self.n_panels }

    #[inline(always)]
    pub fn get_panel_stride( &self ) -> usize { self.panel_stride }
    
    #[inline(always)]
    pub fn get_panel_h( &self ) -> usize { PH::to_usize() }

    #[inline(always)]
    pub unsafe fn get_buffer( &self ) -> *const T { 
        let panel_h = PH::to_usize();
        self.buffer.offset((self.off_panel*self.panel_stride + self.off_x*panel_h) as isize)
    }
    
    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        let panel_h = PH::to_usize();
        self.buffer.offset((self.off_panel*self.panel_stride + self.off_x*panel_h) as isize)
    }

    #[inline(always)]
    pub unsafe fn get_panel( &mut self, id: usize ) -> *mut T {
        self.buffer.offset((self.off_panel + id * self.panel_stride) as isize)
    }
}
impl<T: Scalar, PH: Unsigned> Mat<T> for RowPanelMatrix<T, PH> {
    #[inline(always)]
    fn get( &self, y: usize, x:usize ) -> T {
        let panel_h = PH::to_usize();
        let panel_id = y / panel_h;
        let panel_index  = y % panel_h;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (x + self.off_x) * panel_h + panel_index;
        unsafe{
            ptr::read( self.buffer.offset(elem_index as isize ) )
        }
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x:usize, alpha: T) {
        let panel_h = PH::to_usize();
        let panel_id = y / panel_h;
        let panel_index  = y % panel_h;
        let elem_index = (panel_id + self.off_panel) * self.panel_stride + (x + self.off_x) * panel_h + panel_index;

        unsafe{
            ptr::write( self.buffer.offset(elem_index as isize ), alpha )
        }
    }
    /*
    #[inline(always)]
    fn height( &self ) -> usize{ self.h }
    #[inline(always)]
    fn width( &self ) -> usize{ self.w }
    */
    #[inline(always)]
    fn off_y( &self ) -> usize { 
        let panel_h = PH::to_usize();
        self.off_panel * panel_h 
    }
    #[inline(always)]
    fn off_x( &self ) -> usize { self.off_x }
    /*
    #[inline(always)]
    fn set_height( &mut self, h: usize ) { self.h = h; }
    #[inline(always)]
    fn set_width( &mut self, w: usize ) { self.w = w; }
    */
    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) { 
        let panel_h = PH::to_usize();
        if off_y % panel_h != 0 {
            println!("{} {}", off_y, panel_h);
            panic!("Illegal partitioning within RowPanelMatrix!");
        }
        self.off_panel = off_y / panel_h;
    }
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
    fn set_logical_h_padding( &mut self, h_pad: usize ) { self.logical_h_padding = h_pad }
    #[inline(always)]
    fn get_logical_h_padding( &self ) -> usize { self.logical_h_padding }
    #[inline(always)]
    fn set_logical_w_padding( &mut self, w_pad: usize ) { self.logical_w_padding = w_pad }
    #[inline(always)]
    fn get_logical_w_padding( &self ) -> usize { self.logical_w_padding }
    
    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
        RowPanelMatrix{ iter_h: self.iter_h, iter_w: self.iter_w,
                        logical_h_padding: self.logical_h_padding,
                        logical_w_padding: self.logical_w_padding,
                        off_x: self.off_x, off_panel: self.off_panel,
                        n_panels: self.n_panels,
                        panel_stride: self.panel_stride, 
                        buffer: self.buffer, 
                        capacity: self.capacity,
                        _pht: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias( &self, thr: &ThreadInfo<T> ) -> Self {
        let buf = thr.broadcast( self.buffer );

        RowPanelMatrix{ iter_h: self.iter_h, iter_w: self.iter_w,
                        logical_h_padding: self.logical_h_padding,
                        logical_w_padding: self.logical_w_padding,
                        off_x: self.off_x, off_panel: self.off_panel,
                        n_panels: self.n_panels,
                        panel_stride: self.panel_stride, 
                        buffer: buf, 
                        capacity: self.capacity,
                        _pht: PhantomData }
    }
}
impl<T:Scalar, PH: Unsigned> Drop for RowPanelMatrix<T, PH> {
    fn drop(&mut self) {
        unsafe{ 
            heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
        }
    }
}
unsafe impl<T:Scalar, PH: Unsigned> Send for RowPanelMatrix<T, PH> {}

impl<T:Scalar, PH: Unsigned> ResizeableBuffer<T> for RowPanelMatrix<T, PH> {
    #[inline(always)]
    fn empty() -> Self {
        RowPanelMatrix::new(0,0)
    }
    #[inline(always)]
    fn resize_to_fit( &mut self, other: &Mat<T> ) {
        if self.off_x != 0 || self.off_panel != 0 { panic!("can't resize a submatrix!"); }
        let panel_h = PH::to_usize();

        let mut new_n_panels = other.height() / panel_h;
        if other.height() % panel_h != 0 { new_n_panels += 1; }

        let mut req_capacity = new_n_panels * panel_h * other.width();
        let old_capacity = self.capacity;
        if req_capacity > old_capacity {
            unsafe {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * old_capacity, 4096);
                let newbuf = heap::allocate( req_capacity * mem::size_of::<T>(), 4096 );
                self.buffer = newbuf as *mut _;
                self.capacity = req_capacity;
            }
        }

        self.iter_h = other.iter_height();
        self.iter_w = other.iter_width();
        self.logical_h_padding = other.get_logical_h_padding();
        self.logical_w_padding = other.get_logical_w_padding();

        self.n_panels = new_n_panels;
        self.panel_stride = panel_h*other.width();
    }
}
