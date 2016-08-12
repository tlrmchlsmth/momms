extern crate alloc;

use thread::ThreadInfo;
use typenum::Unsigned;
use self::alloc::heap;
use matrix::{Scalar,Mat,ResizableBuffer};

use core::marker::PhantomData;
use core::mem;
use core::ptr::{self};

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

    is_alias: bool,

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
                           is_alias: false,
                           _pwt: PhantomData }
        }
    }
    

    #[inline(always)]
    pub fn get_n_panels( &self ) -> usize { self.n_panels }

    #[inline(always)]
    pub fn get_panel_stride( &self ) -> usize { self.panel_stride }

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
                           is_alias: true,
                           _pwt: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias( &mut self, thr: &ThreadInfo<T> ) {
        let buf = thr.broadcast( self.buffer );
        self.is_alias = true;
        self.buffer = buf;
    }
}
impl<T:Scalar, PW: Unsigned> Drop for ColumnPanelMatrix<T, PW> {
    fn drop(&mut self) {
        if !self.is_alias {
            unsafe {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
            }
        }
    }
}

unsafe impl<T:Scalar, PW: Unsigned> Send for ColumnPanelMatrix<T, PW> {}

impl<T:Scalar, PW: Unsigned> ResizableBuffer<T> for ColumnPanelMatrix<T, PW> {
    #[inline(always)]
    fn empty() -> Self {
        ColumnPanelMatrix::new(0,0)
    }
    #[inline(always)]
    fn capacity(&self) -> usize { self.capacity }
    #[inline(always)]
    fn set_capacity(&mut self, capacity: usize) { self.capacity = capacity; }
    #[inline(always)]
    fn capacity_for(other: &Mat<T>) -> usize {
        let new_n_panels = (other.width()-1) / PW::to_usize() + 1;
        new_n_panels * PW::to_usize() * other.height()
    }
    #[inline(always)]
    fn aquire_buffer_for(&mut self, req_capacity: usize) {
        if req_capacity > self.capacity {
            unsafe {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
                self.buffer = heap::allocate( req_capacity * mem::size_of::<T>(), 4096 ) as *mut _;
                self.capacity = req_capacity;
            }
        }
    }
    #[inline(always)]
    fn resize_to( &mut self, other: &Mat<T> ) {
        if self.off_y != 0 || self.off_panel != 0 { panic!("can't resize a submatrix!"); }

        self.iter_h = other.iter_height();
        self.iter_w = other.iter_width();
        self.logical_h_padding = other.get_logical_h_padding();
        self.logical_w_padding = other.get_logical_w_padding();

        self.n_panels = (other.width()-1) / PW::to_usize() + 1;
        self.panel_stride = PW::to_usize()*other.height();
    }
}

