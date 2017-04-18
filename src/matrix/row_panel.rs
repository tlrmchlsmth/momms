extern crate alloc;

use thread_comm::ThreadInfo;
use typenum::Unsigned;
use self::alloc::heap;
use matrix::{Scalar,Mat,ResizableBuffer};
use super::view::{MatrixView};

use core::marker::PhantomData;
use core::{mem};
use core::ptr::{self};

use composables::{AlgorithmStep};

pub struct RowPanelMatrix<T: Scalar, PH: Unsigned> {
    y_views: Vec<MatrixView>, //offset is in # of panels
    x_views: Vec<MatrixView>,

    panel_stride: usize,
    buffer: *mut T,
    capacity: usize,
    is_alias: bool,

    _pht: PhantomData<PH>,
}
impl<T: Scalar, PH: Unsigned> RowPanelMatrix<T,PH> {
    pub fn new( h: usize, w: usize ) -> RowPanelMatrix<T,PH> {
        assert!(mem::size_of::<T>() != 0, "Matrix can't handle ZSTs");
    
        //Figure out the number of panels
        let panel_h = PH::to_usize();
        let mut n_panels = h / panel_h;
        if n_panels * panel_h < h {
            n_panels += 1;
        }
        let capacity = (n_panels+1) * panel_h * w; //Extra panel for ``preloading'' in ukernel

        let mut y_views : Vec<MatrixView> = Vec::with_capacity( 16 );
        let mut x_views : Vec<MatrixView> = Vec::with_capacity( 16 );
        y_views.push(MatrixView{ offset: 0, padding: 0, iter_size: h }); 
        x_views.push(MatrixView{ offset: 0, padding: 0, iter_size: w }); 

        unsafe { 
            let ptr = heap::allocate( capacity * mem::size_of::<T>(), 4096 );

            RowPanelMatrix{ y_views: y_views, x_views: x_views,
                               panel_stride: panel_h*w, 
                               buffer: ptr as *mut _, capacity: capacity,
                               is_alias: false,
                               _pht: PhantomData }
        }
    }
    

    #[inline(always)]
    pub fn get_panel_stride( &self ) -> usize { self.panel_stride }

    #[inline(always)]
    pub unsafe fn get_buffer( &self ) -> *const T { 
        let panel_h = PH::to_usize();
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        self.buffer.offset((x_view.offset*panel_h + y_view.offset*self.panel_stride) as isize)
    }

    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T {
        let panel_h = PH::to_usize();
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        self.buffer.offset((x_view.offset*panel_h + y_view.offset*self.panel_stride) as isize)
    }

    #[inline(always)]
    pub unsafe fn get_panel( &mut self, id: usize ) -> *mut T {
        let y_view = self.y_views.last().unwrap();
        self.buffer.offset(((y_view.offset + id)*self.panel_stride) as isize)
    }
}
impl<T: Scalar, PH: Unsigned> Mat<T> for RowPanelMatrix<T, PH> {
    #[inline(always)]
    fn get( &self, y: usize, x: usize ) -> T {
        let panel_h = PH::to_usize();
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        let panel_id = y / panel_h;
        let panel_index  = y % panel_h;
        let elem_index = (panel_id + y_view.offset) * self.panel_stride + (x + x_view.offset) * panel_h + panel_index;
        unsafe{
            ptr::read( self.buffer.offset(elem_index as isize ) )
        }
    }
    #[inline(always)]
    fn set( &mut self, y: usize, x: usize, alpha: T) {
        let panel_h = PH::to_usize();
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        let panel_id = y / panel_h;
        let panel_index  = y % panel_h;
        let elem_index = (panel_id + y_view.offset) * self.panel_stride + (x + x_view.offset) * panel_h + panel_index;
        unsafe{
            ptr::write( self.buffer.offset(elem_index as isize ), alpha );
        }
    }
    
    #[inline(always)]
    fn off_y( &self ) -> usize { 
        self.y_views.last().unwrap().offset * PH::to_usize()
    }
    #[inline(always)]
    fn off_x( &self ) -> usize { 
        self.x_views.last().unwrap().offset
    }

    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) {
        let panel_h = PH::to_usize();
        if off_y % panel_h != 0 {
            println!("{} {}", off_y, panel_h);
            panic!("Illegal partitioning within RowPanelMatrix!");
        }

        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.offset = off_y / panel_h;
    }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) { 
        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.offset = off_x; 
    }
    #[inline(always)]
    fn add_off_y( &mut self, start: usize ) { 
        let off = self.off_y();
        self.set_off_y(start + off);
    }   
    #[inline(always)]
    fn add_off_x( &mut self, start: usize ) { 
        let off = self.off_x();
        self.set_off_x(start + off);
    }


    #[inline(always)]
    fn iter_height( &self ) -> usize {
        let y_view = self.y_views.last().unwrap();
        y_view.iter_size
    }
    #[inline(always)]
    fn iter_width( &self ) -> usize {
        let x_view = self.x_views.last().unwrap();
        x_view.iter_size
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
        let y_view = self.y_views.last().unwrap();
        y_view.padding
    }
    #[inline(always)]
    fn logical_w_padding( &self ) -> usize {
        let x_view = self.x_views.last().unwrap();
        x_view.padding
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

    fn push_x_view( &mut self, blksz: usize ) -> usize {
        let (zoomed_view, uz_iter_size) = {
            let uz_view = self.x_views.last().unwrap();
            let (z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(0, blksz);
            (MatrixView{ offset: uz_view.offset, padding: z_padding, iter_size: z_iter_size }, uz_view.iter_size)
        };
        self.x_views.push(zoomed_view);
        uz_iter_size
    }
    
    fn push_y_view( &mut self, blksz: usize ) -> usize{
        let (zoomed_view, uz_iter_size) = {
            let uz_view = self.y_views.last().unwrap();
            let (z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(0, blksz);
            (MatrixView{ offset: uz_view.offset, padding: z_padding, iter_size: z_iter_size }, uz_view.iter_size)
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

    fn slide_x_view_to( &mut self, x: usize, blksz: usize ) {
        let view_len = self.x_views.len();
        debug_assert!( view_len >= 2 );

        let uz_view = self.x_views[view_len-2];
        let(z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(x, blksz);

        let mut z_view = self.x_views.last_mut().unwrap();
        z_view.iter_size = z_iter_size;
        z_view.padding = z_padding;
        z_view.offset = uz_view.offset + x;
    }
    
    fn slide_y_view_to( &mut self, y: usize, blksz: usize ) {
        let view_len = self.y_views.len();
        debug_assert!( view_len >= 2 );

        let uz_view = self.y_views[view_len-2];
        let(z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(y, blksz);

        let mut z_view = self.y_views.last_mut().unwrap();
        z_view.iter_size = z_iter_size;
        z_view.padding = z_padding;
        z_view.offset = uz_view.offset + y / PH::to_usize();
    }
    
    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
        let x_view = self.x_views.last().unwrap();
        let y_view = self.y_views.last().unwrap();

        let mut x_views_alias : Vec<MatrixView> = Vec::with_capacity( 16 );
        let mut y_views_alias : Vec<MatrixView> = Vec::with_capacity( 16 );
        x_views_alias.push(MatrixView{ offset: x_view.offset, padding: x_view.offset, iter_size: x_view.iter_size });
        y_views_alias.push(MatrixView{ offset: y_view.offset, padding: y_view.offset, iter_size: y_view.iter_size });

        RowPanelMatrix{ y_views: y_views_alias, x_views: x_views_alias,
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
    fn empty(_: AlgorithmStep, _: AlgorithmStep, _: &Vec<AlgorithmStep>) -> Self {
        RowPanelMatrix::new(0,0)
    }
    #[inline(always)]
    fn capacity(&self) -> usize { self.capacity }
    #[inline(always)]
    fn set_capacity(&mut self, capacity: usize) { self.capacity = capacity; }
    #[inline(always)]
    fn capacity_for(other: &Mat<T>, _: AlgorithmStep, _: AlgorithmStep, _: &Vec<AlgorithmStep>) -> usize {
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
    fn resize_to( &mut self, other: &Mat<T>, _: AlgorithmStep, _: AlgorithmStep, _: &Vec<AlgorithmStep> ) {
        debug_assert!( self.y_views.len() == 1, "Can't resize a submatrix!");
        let mut y_view = self.y_views.last_mut().unwrap();
        let mut x_view = self.x_views.last_mut().unwrap();

        y_view.iter_size = other.iter_height();
        x_view.iter_size = other.iter_width();
        y_view.padding = other.logical_h_padding();
        x_view.padding = other.logical_w_padding();
        self.panel_stride = PH::to_usize()*other.width();
    }
}
