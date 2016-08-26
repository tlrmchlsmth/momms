extern crate alloc;

use thread_comm::ThreadInfo;
use typenum::Unsigned;
use self::alloc::heap;
use matrix::{Scalar,Mat,ResizableBuffer};

use core::marker::PhantomData;
use core::mem;
use core::ptr::{self};

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

    is_alias: bool,

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
        
        let capacity = (n_panels+1) * panel_h * w;
        unsafe { 
            let ptr = heap::allocate( capacity * mem::size_of::<T>(), 4096 );

            RowPanelMatrix{ iter_h: h, iter_w: w,
                            logical_h_padding: 0, logical_w_padding: 0,
                            off_x: 0, off_panel: 0,
                            n_panels : n_panels, panel_stride: panel_h*w, 
                            buffer: ptr as *mut _, capacity: capacity,
                            is_alias: false,
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
    fn logical_h_padding( &self ) -> usize { self.logical_h_padding }
    #[inline(always)]
    fn set_logical_w_padding( &mut self, w_pad: usize ) { self.logical_w_padding = w_pad }
    #[inline(always)]
    fn logical_w_padding( &self ) -> usize { self.logical_w_padding }
    
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
                        is_alias: true,
                        _pht: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias( &mut self, thr: &ThreadInfo<T> ) {
        let buf = thr.broadcast( self.buffer );
        self.is_alias = true;
        self.buffer = buf;
    }
}
impl<T:Scalar, PH: Unsigned> Drop for RowPanelMatrix<T, PH> {
    fn drop(&mut self) {
        if !self.is_alias {
            unsafe { 
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
            }
        }
    }
}
unsafe impl<T:Scalar, PH: Unsigned> Send for RowPanelMatrix<T, PH> {}

impl<T:Scalar, PH: Unsigned> ResizableBuffer<T> for RowPanelMatrix<T, PH> {
    #[inline(always)]
    fn empty() -> Self {
        RowPanelMatrix::new(0,0)
    }
    #[inline(always)]
    fn capacity(&self) -> usize { self.capacity }
    #[inline(always)]
    fn set_capacity(&mut self, capacity: usize) { self.capacity = capacity; }
    #[inline(always)]
    fn capacity_for(other: &Mat<T>) -> usize {
        if other.height() <= 0 || other.width() <= 0 {
            0
        } else {
            let new_n_panels = (other.height()-1) / PH::to_usize() + 1;
            (new_n_panels + 1) * PH::to_usize() * other.width()
        }
    }
    #[inline(always)]
    fn aquire_buffer_for(&mut self, req_capacity: usize) {
        let req_padded_capacity = req_capacity;
        if req_padded_capacity > self.capacity {
            unsafe {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
                self.buffer = heap::allocate( req_padded_capacity * mem::size_of::<T>(), 4096 ) as *mut _;
                self.capacity = req_padded_capacity;
            }
        }
    }
    #[inline(always)]
    fn resize_to( &mut self, other: &Mat<T> ) {
        if self.off_x != 0 || self.off_panel != 0 { panic!("can't resize a submatrix!"); }

        self.iter_h = other.iter_height();
        self.iter_w = other.iter_width();
        self.logical_h_padding = other.logical_h_padding();
        self.logical_w_padding = other.logical_w_padding();
        
        if other.height() <= 0 {
            self.n_panels = 0;
        } else {
            self.n_panels = (other.height()-1) / PH::to_usize() + 1;
        }
        self.panel_stride = PH::to_usize()*other.width();
    }
}
