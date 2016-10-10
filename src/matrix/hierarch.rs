extern crate alloc;
extern crate generic_array;

use thread_comm::ThreadInfo;
use typenum::Unsigned;
use self::alloc::heap;
use matrix::{Scalar,Mat,ResizableBuffer};
use super::view::{MatrixView};
use composables::AlgorithmStep;

use core::marker::PhantomData;
use core::mem;
use core::ptr::{self};

/*Notes:
 * This file is for implementing hierarchical matrices
 * These are matrices where the data is physically arranged hierarchically for spatial locality.
 * 
 * This hierarchy can be described as a list of directions, blocksizes, and strides
 * At the bottom of this hierarchy there are leaf blocks. 
 * Matrices always get padded s.t. they can be divided into an integer # of rows and columns of leaf blocks.
 */

#[derive(Clone)]
pub struct HierarchyNode {
    pub stride: usize, //This is the stride between submatrices
    pub blksz: usize,
}

//I need to be able to specify everything about a hierarch matrix in its type!
//Otherwise, how does the packer set it up??
//It can't know to "add to x hierarchy"!
//This may involve passing a gemm algorithm as a type to generalize over
//Notice that the type of that algorithm can't be the same type that actually runs the
//gemm with hierarchiccal matrices! Because that type depends on the type of this and we can't have
//cyclical types.

//LRS is leaf row stride, LCS is leaf column stride
pub struct Hierarch<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned>{ 
    //We pop from the hierarchy nodes, and push into the view nodes (with some details added)

    //These are the view stacks just like the other Matrices
    y_views: Vec<MatrixView>,
    x_views: Vec<MatrixView>,

    //These describe the hierarchy of the current view
    y_hierarchy: Vec<HierarchyNode>,
    x_hierarchy: Vec<HierarchyNode>,

    //Tracks current position within x and y hierarchies
    yh_index: usize,
    xh_index: usize,

    buffer: *mut T,
    capacity: usize,
    is_alias: bool,
    //_ht:   PhantomData<H>,
    _lht:  PhantomData<LH>,
    _lwt:  PhantomData<LW>,
    _lrst: PhantomData<LRS>,
    _lcst: PhantomData<LCS>,
}

impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> Hierarch<T, LH, LW, LRS, LCS> {
    fn get_iota( hierarchy: &[AlgorithmStep], dimension_matcher: AlgorithmStep ) -> usize{
        use composables::AlgorithmStep::*;
        match( dimension_matcher, hierarchy[hierarchy.len()-1] ) {
            (M{bsz: _}, M{bsz}) => {bsz},
            (N{bsz: _}, N{bsz}) => {bsz},
            (K{bsz: _}, K{bsz}) => {bsz},
            _ => Self::get_iota( &hierarchy[0..hierarchy.len()-1], dimension_matcher ),
        }
    }
    fn parse_input_hierarchy( h: usize, w: usize,
                              hier: &[AlgorithmStep], y_step: AlgorithmStep, x_step: AlgorithmStep ) 
        -> (Vec<HierarchyNode>, Vec<HierarchyNode>) {

        let mut y_hierarchy = Vec::with_capacity(16);
        let mut x_hierarchy = Vec::with_capacity(16);

        let mut h_tracker = h;
        let mut w_tracker = w;
        
        for index in (0..hier.len()).rev() {
            use composables::AlgorithmStep::*;
            match (x_step, hier[index]) {
                (M{bsz:_}, M{bsz}) => {
                    x_hierarchy.push(HierarchyNode{stride: h_tracker * bsz, blksz: bsz});
                    w_tracker = bsz;},
                (N{bsz:_}, N{bsz}) => {
                    x_hierarchy.push(HierarchyNode{stride: h_tracker * bsz, blksz: bsz});
                    w_tracker = bsz;},
                (K{bsz:_}, K{bsz}) => {
                    x_hierarchy.push(HierarchyNode{stride: h_tracker * bsz, blksz: bsz});
                    w_tracker = bsz;},
                _ => {},
            };
            match (y_step, hier[index]) {
                (M{bsz:_}, M{bsz}) => {
                    y_hierarchy.push(HierarchyNode{stride: w_tracker * bsz, blksz: bsz});
                    h_tracker = bsz;},
                (N{bsz:_}, N{bsz}) => {
                    y_hierarchy.push(HierarchyNode{stride: w_tracker * bsz, blksz: bsz});
                    h_tracker = bsz;},
                (K{bsz:_}, K{bsz}) => {
                    y_hierarchy.push(HierarchyNode{stride: w_tracker * bsz, blksz: bsz});
                    h_tracker = bsz;},
                _ => {},
            };
        }
        y_hierarchy.push(HierarchyNode{stride:0, blksz:0});
        x_hierarchy.push(HierarchyNode{stride:0, blksz:0});

        (y_hierarchy, x_hierarchy)

    }
    pub fn new( h: usize, w: usize, 
                hier: &[AlgorithmStep], y_step: AlgorithmStep, x_step: AlgorithmStep ) 
        -> Hierarch<T,LH,LW,LRS,LCS> {
        assert!(mem::size_of::<T>() != 0, "Matrix can't handle ZSTs");

        //Setup Views stack
        let mut y_views : Vec<MatrixView> = Vec::with_capacity(16);
        let mut x_views : Vec<MatrixView> = Vec::with_capacity(16);
        y_views.push(MatrixView{ offset: 0, padding: 0, iter_size: h });
        x_views.push(MatrixView{ offset: 0, padding: 0, iter_size: w });

        //Fill up hierarchy description
        let y_iota = Self::get_iota(&hier, y_step);
        let x_iota = Self::get_iota(&hier, x_step);

        let n_blocks_y = (h-1) / y_iota + 1;
        let n_blocks_x = (w-1) / x_iota + 1;
        let h_padded = n_blocks_y * y_iota;
        let w_padded = n_blocks_x * x_iota;

        let (y_hierarchy, x_hierarchy) = Self::parse_input_hierarchy( h_padded, w_padded, &hier, y_step, x_step );
        let yh_index = 0;
        let xh_index = 0;

        //Figure out buffer and capacity
        let (ptr, capacity) = 
            if h == 0 || w == 0 {
                unsafe { 
                    (heap::allocate( 0 * mem::size_of::<T>(), 4096 ), 0)
                }
            } else {
                //Figure out the number of top-level blocks in each direction
                let capacity = (n_blocks_y + 1) * y_iota * (n_blocks_x + 1) * x_iota;

                //Allocate Buffer
                unsafe { 
                    (heap::allocate( capacity * mem::size_of::<T>(), 4096 ), capacity)
                }
            };

      
        //Return
        Hierarch{ y_views: y_views, x_views: x_views,
                  y_hierarchy: y_hierarchy, x_hierarchy: x_hierarchy,
                  yh_index: yh_index, xh_index: xh_index,  
                  buffer: ptr as *mut _, capacity: capacity,
                  is_alias: false,
                  _lht: PhantomData, _lwt: PhantomData,
                  _lrst: PhantomData, _lcst: PhantomData }
    }

    #[inline(always)]
    pub unsafe fn get_buffer( &self ) -> *const T { 
        self.buffer.offset(self.get_offset(0,0))
    }   

    #[inline(always)]
    pub unsafe fn get_mut_buffer( &mut self ) -> *mut T { 
        self.buffer.offset(self.get_offset(0,0))
    }  

    //Helper functions to drill down the cache hierarchy to get the physical location of a matrix
    //element
    #[inline(always)]
    pub fn get_offset_y(&self, y: usize) -> isize {
        let mut y_off = self.y_views.last().unwrap().offset; // Tracks the physical address of the row y
        let mut y_index = y;                                 // Tracks the logical address within the current block of the row y

        for index in self.yh_index..self.y_hierarchy.len()-1 {
            let block_index = y_index / self.y_hierarchy[index].blksz;
            y_off += block_index * self.y_hierarchy[index].stride;
            y_index -= block_index * self.y_hierarchy[index].blksz;
        }

        //Handle the leaf node
        (y_off + y_index * LRS::to_usize()) as isize
    }
    #[inline(always)]
    pub fn get_offset_x(&self, x: usize) -> isize {
        let mut x_off = self.x_views.last().unwrap().offset; // Tracks the physical address of the row x
        let mut x_index = x;                                 // Tracks the logical address within the current block of the row x

        for index in self.xh_index..self.x_hierarchy.len()-1 {
            let block_index = x_index / self.x_hierarchy[index].blksz;
            x_off += block_index * self.x_hierarchy[index].stride;
            x_index -= block_index * self.x_hierarchy[index].blksz;
        }
        
        //Handle the leaf node
        (x_off + x_index * LCS::to_usize()) as isize
    }
    #[inline(always)]
    pub fn get_offset(&self, y: usize, x: usize) -> isize {
        self.get_offset_y(y) + self.get_offset_x(x)
    }
}
impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> Mat<T> for
    Hierarch<T, LH, LW, LRS, LCS> {
    #[inline(always)]
    fn get(&self, y: usize, x: usize) -> T {
        unsafe{
            ptr::read(self.buffer.offset(self.get_offset(y,x)))
        }
    }
    #[inline(always)]
    fn set(&mut self, y: usize, x: usize, alpha: T) {
        unsafe{
            ptr::write(self.buffer.offset(self.get_offset(y,x)), alpha);
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

    #[inline(always)]
    fn set_off_y( &mut self, off_y: usize ) {
        let iota = self.y_hierarchy[self.yh_index].blksz;
        if off_y % iota != 0 {
            panic!("Illegal partitioning within Hierarch!");
        }

        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.offset = off_y;
    }
    #[inline(always)]
    fn set_off_x( &mut self, off_x: usize ) {
        let iota = self.x_hierarchy[self.xh_index].blksz;
        if off_x % iota != 0 {
            panic!("Illegal partitioning within Hierarch!");
        }

        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.offset = off_x;
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

    #[inline(always)]
    fn push_y_view( &mut self, blksz: usize ) -> usize{
        let iota = self.y_hierarchy[self.yh_index].blksz;
        debug_assert!(iota == blksz, "iota {}, blksz {} {} ", iota, blksz, self.yh_index);

        let (zoomed_view, uz_iter_size) = {
            let uz_view = self.y_views.last().unwrap();
            let (z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(0, blksz);
            (MatrixView{ offset: uz_view.offset, padding: z_padding, iter_size: z_iter_size }, uz_view.iter_size)
        };
        self.y_views.push(zoomed_view);
        self.yh_index += 1;
        uz_iter_size
    }
    #[inline(always)]
    fn push_x_view( &mut self, blksz: usize ) -> usize {
        let iota = self.x_hierarchy[self.xh_index].blksz;
        debug_assert!(iota == blksz);

        let (zoomed_view, uz_iter_size) = {
            let uz_view = self.x_views.last().unwrap();
            let (z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(0, blksz);
            (MatrixView{ offset: uz_view.offset, padding: z_padding, iter_size: z_iter_size }, uz_view.iter_size)
        };
        self.x_views.push(zoomed_view);
        self.xh_index += 1;
        uz_iter_size
    }

    #[inline(always)]
    fn pop_y_view( &mut self ) {
        debug_assert!( self.y_views.len() >= 2 );
        self.y_views.pop();
        self.yh_index -= 1;
    }
    #[inline(always)]
    fn pop_x_view( &mut self ) {
        debug_assert!( self.x_views.len() >= 2 );
        self.x_views.pop();
        self.xh_index -= 1;
    }
    #[inline(always)]
    fn slide_y_view_to( &mut self, y: usize, blksz: usize ) {
        let view_len = self.y_views.len();
        debug_assert!( view_len >= 2 );
        let iota = self.y_hierarchy[self.yh_index-1].blksz;
        debug_assert!( y % iota == 0 );

        let uz_view = self.y_views[view_len-2];
        let(z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(y, blksz);

        let mut z_view = self.y_views.last_mut().unwrap();
        z_view.iter_size = z_iter_size;
        z_view.padding = z_padding;
        z_view.offset = uz_view.offset + (y / iota) * self.y_hierarchy[self.yh_index-1].stride;
    }
    #[inline(always)]
    fn slide_x_view_to( &mut self, x: usize, blksz: usize ) {
        let view_len = self.x_views.len();
        debug_assert!( view_len >= 2 );
        let iota = self.x_hierarchy[self.xh_index-1].blksz;
        debug_assert!( x % iota == 0 );

        let uz_view = self.x_views[view_len-2];
        let(z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(x, blksz);

        let mut z_view = self.x_views.last_mut().unwrap();
        z_view.iter_size = z_iter_size;
        z_view.padding = z_padding;
        z_view.offset = uz_view.offset + (x / iota) * self.x_hierarchy[self.xh_index-1].stride;
    }

    #[inline(always)]
    unsafe fn make_alias( &self ) -> Self {
        //Setup alias's view
        let x_view = self.x_views.last().unwrap();
        let y_view = self.y_views.last().unwrap();

        let mut x_views_alias : Vec<MatrixView> = Vec::with_capacity( 16 );
        let mut y_views_alias : Vec<MatrixView> = Vec::with_capacity( 16 );
        x_views_alias.push(MatrixView{ offset: x_view.offset, padding: x_view.offset, iter_size: x_view.iter_size });
        y_views_alias.push(MatrixView{ offset: y_view.offset, padding: y_view.offset, iter_size: y_view.iter_size });

        //Setup alias's hierarchy
        let x_hierarchy = self.x_hierarchy.clone();
        let y_hierarchy = self.y_hierarchy.clone();

        Hierarch{ y_views: y_views_alias, x_views: x_views_alias,
                  y_hierarchy: y_hierarchy, x_hierarchy: x_hierarchy,
                  yh_index: self.yh_index, xh_index: self.xh_index,  
                  buffer: self.buffer,
                  capacity: self.capacity,
                  is_alias: true,
                  _lht: PhantomData, _lwt: PhantomData,
                  _lrst: PhantomData, _lcst: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias( &mut self, thr: &ThreadInfo<T> ) {
        let buf = thr.broadcast( self.buffer );
        self.is_alias = true;
        self.buffer = buf;
    }
}
impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> Drop for Hierarch<T, LH, LW, LRS, LCS> {
    fn drop(&mut self) {
        if !self.is_alias {
            unsafe {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
            }
        }
    }
}
unsafe impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> Send for Hierarch<T, LH, LW, LRS, LCS> {}

impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> ResizableBuffer<T>
     for Hierarch<T, LH, LW, LRS, LCS> {
    #[inline(always)]
    fn empty() -> Self {
        panic!("Not implemented!");
//        Hierarch::new(0,0)
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
            panic!("capacity for other not implemented!");
/*            let n_blocks_y = (h - 1) / self.y_hierarchy[self.yh_index].blksz + 1;
            let n_blocks_x = (w - 1) / self.x_hierarchy[self.xh_index].blksz + 1;
            (n_blocks_y + 1) * self.y_hierarchy[self.yh_index].blksz * (n_blocks_x + 1) * self.x_hierarchy[self.xh_index].blksz*/
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
        panic!("resize_to is not implemented for hierarchies yet!");
    }
}




