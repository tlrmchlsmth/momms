extern crate alloc;

use thread_comm::ThreadInfo;
use typenum::Unsigned;
use self::alloc::heap;
use matrix::{Scalar,Mat,ResizableBuffer,RoCM};
use super::view::{MatrixView};
use composables::AlgorithmStep;

use core::marker::PhantomData;
use core::{mem,ptr};

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
    alpha: T,

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
    pub fn get_y_hierarchy(&self) -> &[HierarchyNode] {
        &self.y_hierarchy[self.yh_index..self.y_hierarchy.len()]
    }
    pub fn get_x_hierarchy(&self) -> &[HierarchyNode] {
        &self.x_hierarchy[self.xh_index..self.x_hierarchy.len()]
    }
    fn get_top_level_dim_size(hierarchy: &[AlgorithmStep], dimension_matcher: AlgorithmStep) -> Option<usize> {
        use composables::AlgorithmStep::*;
        if hierarchy.len() == 0 { None } else {
            match (dimension_matcher, hierarchy[hierarchy.len()-1]) {
                (M{bsz: _}, M{bsz}) => {Some(bsz)},
                (N{bsz: _}, N{bsz}) => {Some(bsz)},
                (K{bsz: _}, K{bsz}) => {Some(bsz)},
                _ => Self::get_top_level_dim_size(&hierarchy[0..hierarchy.len()-1], dimension_matcher),
            } 
        }
    }
    fn parse_input_hierarchy(h: usize, w: usize,
                              hier: &[AlgorithmStep], y_step: AlgorithmStep, x_step: AlgorithmStep) 
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
        y_hierarchy.push(HierarchyNode{stride:1, blksz:1});
        x_hierarchy.push(HierarchyNode{stride:1, blksz:1});

        (y_hierarchy, x_hierarchy)

    }
    pub fn new(h: usize, w: usize, 
                hier: &[AlgorithmStep], y_step: AlgorithmStep, x_step: AlgorithmStep) 
        -> Hierarch<T,LH,LW,LRS,LCS> {
        assert!(mem::size_of::<T>() != 0, "Matrix can't handle ZSTs");

        //Setup Views stack
        let mut y_views : Vec<MatrixView> = Vec::with_capacity(16);
        let mut x_views : Vec<MatrixView> = Vec::with_capacity(16);
        y_views.push(MatrixView{ offset: 0, padding: 0, iter_size: h });
        x_views.push(MatrixView{ offset: 0, padding: 0, iter_size: w });

        //Fill up hierarchy description
        let y_tlds = match Self::get_top_level_dim_size(&hier, y_step) {
            Some(a) => a,
            None => LH::to_usize(),
        };
        let x_tlds = match Self::get_top_level_dim_size(&hier, x_step) {
            Some(a) => a,
            None => LW::to_usize(),
        };
    
        let n_blocks_y = if h == 0 {1} else {(h-1) / y_tlds + 1};
        let n_blocks_x = if w == 0 {1} else {(w-1) / x_tlds + 1};
        let h_padded = n_blocks_y * y_tlds;
        let w_padded = n_blocks_x * x_tlds;

        let (y_hierarchy, x_hierarchy) = Self::parse_input_hierarchy(h_padded, w_padded, &hier, y_step, x_step);
        let yh_index = 0;
        let xh_index = 0;

        //Figure out buffer and capacity
        let (ptr, capacity) = 
            if h == 0 || w == 0 {
                unsafe { 
                    (heap::allocate(0 * mem::size_of::<T>(), 4096), 0)
                }
            } else {
                //Figure out the number of top-level blocks in each direction
                let capacity = (n_blocks_y + 1) * y_tlds * (n_blocks_x + 1) * x_tlds;

                //Allocate Buffer
                unsafe { 
                    (heap::allocate(capacity * mem::size_of::<T>(), 4096), capacity)
                }
            };

      
        //Return
        Hierarch{ alpha: T::one(),
                  y_views: y_views, x_views: x_views,
                  y_hierarchy: y_hierarchy, x_hierarchy: x_hierarchy,
                  yh_index: yh_index, xh_index: xh_index,  
                  buffer: ptr as *mut _, capacity: capacity,
                  is_alias: false,
                  _lht: PhantomData, _lwt: PhantomData,
                  _lrst: PhantomData, _lcst: PhantomData }
    }
/*
    #[inline(always)]
    pub unsafe fn get_buffer(&self) -> *const T { 
        let y_off = self.y_views.last().unwrap().offset;
        let x_off = self.x_views.last().unwrap().offset;
        self.buffer.offset((y_off + x_off) as isize) 
    }   

    #[inline(always)]
    pub unsafe fn get_mut_buffer(&mut self) -> *mut T { 
        let y_off = self.y_views.last().unwrap().offset;
        let x_off = self.x_views.last().unwrap().offset;
        self.buffer.offset((y_off + x_off) as isize) 
    }
*/
    #[inline(always)]
    pub fn block_stride_x(&self, level: usize) -> usize {
        self.x_hierarchy[self.xh_index + level].stride
    }
    #[inline(always)]
    pub fn block_stride_y(&self, level: usize) -> usize {
        self.y_hierarchy[self.yh_index + level].stride
    }
      
    //Helper functions to drill down the cache hierarchy to get the physical location of a matrix
    //element
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

    //Used to save and restore offsets
    #[inline(always)]
    fn off_y(&self) -> usize { 
        self.y_views.last().unwrap().offset
    }
    #[inline(always)]
    fn off_x(&self) -> usize { 
        self.x_views.last().unwrap().offset
    }

    #[inline(always)]
    fn set_off_y(&mut self, off_y: usize) {
        debug_assert!(off_y % self.y_hierarchy[self.yh_index].blksz == 0);
        
        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.offset = off_y;
    }
    #[inline(always)]
    fn set_off_x(&mut self, off_x: usize) {
        debug_assert!(off_x % self.x_hierarchy[self.xh_index].blksz == 0);

        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.offset = off_x;
    }

    #[inline(always)]
    fn add_off_y(&mut self, start: usize) {
        debug_assert!(start % self.y_hierarchy[self.yh_index].blksz == 0);

        let n_blocks = start / self.y_hierarchy[self.yh_index].blksz;
        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.offset += n_blocks * self.y_hierarchy[self.yh_index].stride;
    }
    #[inline(always)]
    fn add_off_x(&mut self, start: usize) {
        debug_assert!(start % self.x_hierarchy[self.xh_index].blksz == 0);

        let n_blocks = start / self.x_hierarchy[self.xh_index].blksz;
        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.offset += n_blocks * self.x_hierarchy[self.xh_index].stride;
    }

    #[inline(always)]
    fn iter_height(&self) -> usize {
        let y_view = self.y_views.last().unwrap();
        y_view.iter_size
    }
    #[inline(always)]
    fn iter_width(&self) -> usize {
        let x_view = self.x_views.last().unwrap();
        x_view.iter_size
    }
    #[inline(always)]
    fn set_iter_height(&mut self, iter_h: usize) {
        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.iter_size = iter_h;
    }
    #[inline(always)]
    fn set_iter_width(&mut self, iter_w: usize) {
        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.iter_size = iter_w;
    }

    #[inline(always)]
    fn logical_h_padding(&self) -> usize {
        let y_view = self.y_views.last().unwrap();
        y_view.padding
    }
    #[inline(always)]
    fn logical_w_padding(&self) -> usize {
        let x_view = self.x_views.last().unwrap();
        x_view.padding
    }
    #[inline(always)]
    fn set_logical_h_padding(&mut self, h_pad: usize) {
        let mut y_view = self.y_views.last_mut().unwrap();
        y_view.padding = h_pad
    }

    #[inline(always)]
    fn set_logical_w_padding(&mut self, w_pad: usize) {
        let mut x_view = self.x_views.last_mut().unwrap();
        x_view.padding = w_pad
    }
    #[inline(always)]
    fn set_scalar(&mut self, alpha: T) {
        self.alpha = alpha;
    }
    #[inline(always)]
    fn get_scalar(&self) -> T {
        self.alpha
    }

    fn push_y_view(&mut self, blksz: usize) -> usize{
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
    
    fn push_x_view(&mut self, blksz: usize) -> usize {
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
    fn pop_y_view(&mut self) {
        debug_assert!(self.y_views.len() >= 2);
        self.y_views.pop();
        self.yh_index -= 1;
    }
    #[inline(always)]
    fn pop_x_view(&mut self) {
        debug_assert!(self.x_views.len() >= 2);
        self.x_views.pop();
        self.xh_index -= 1;
    }
    
    fn slide_y_view_to(&mut self, y: usize, blksz: usize) {
        let view_len = self.y_views.len();
        debug_assert!(view_len >= 2);
        let iota = self.y_hierarchy[self.yh_index-1].blksz;
        debug_assert!(y % iota == 0);

        let uz_view = self.y_views[view_len-2];
        let(z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(y, blksz);

        let mut z_view = self.y_views.last_mut().unwrap();
        z_view.iter_size = z_iter_size;
        z_view.padding = z_padding;
        z_view.offset = uz_view.offset + (y / iota) * self.y_hierarchy[self.yh_index-1].stride;
    }

    fn slide_x_view_to(&mut self, x: usize, blksz: usize) {
        let view_len = self.x_views.len();
        debug_assert!(view_len >= 2);
        let iota = self.x_hierarchy[self.xh_index-1].blksz;
        debug_assert!(x % iota == 0);

        let uz_view = self.x_views[view_len-2];
        let(z_iter_size, z_padding) = uz_view.zoomed_size_and_padding(x, blksz);

        let mut z_view = self.x_views.last_mut().unwrap();
        z_view.iter_size = z_iter_size;
        z_view.padding = z_padding;
        z_view.offset = uz_view.offset + (x / iota) * self.x_hierarchy[self.xh_index-1].stride;
    }

    #[inline(always)]
    unsafe fn make_alias(&self) -> Self {
        //Setup alias's view
        let x_view = self.x_views.last().unwrap();
        let y_view = self.y_views.last().unwrap();

        let mut x_views_alias : Vec<MatrixView> = Vec::with_capacity(16);
        let mut y_views_alias : Vec<MatrixView> = Vec::with_capacity(16);
        x_views_alias.push(MatrixView{ offset: x_view.offset, padding: x_view.offset, iter_size: x_view.iter_size });
        y_views_alias.push(MatrixView{ offset: y_view.offset, padding: y_view.offset, iter_size: y_view.iter_size });

        //Setup alias's hierarchy
        let x_hierarchy = self.x_hierarchy.clone();
        let y_hierarchy = self.y_hierarchy.clone();

        Hierarch{ alpha: T::one(),
                  y_views: y_views_alias, x_views: x_views_alias,
                  y_hierarchy: y_hierarchy, x_hierarchy: x_hierarchy,
                  yh_index: self.yh_index, xh_index: self.xh_index,  
                  buffer: self.buffer,
                  capacity: self.capacity,
                  is_alias: true,
                  _lht: PhantomData, _lwt: PhantomData,
                  _lrst: PhantomData, _lcst: PhantomData }
    }

    #[inline(always)]
    unsafe fn send_alias(&mut self, thr: &ThreadInfo<T>) {
        let buf = thr.broadcast(self.buffer);
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
    fn empty(y_hier_label: AlgorithmStep, x_hier_label: AlgorithmStep, hier: &Vec<AlgorithmStep>) -> Self {
        Hierarch::new(0,0, hier, y_hier_label, x_hier_label)
    }
    #[inline(always)]
    fn capacity(&self) -> usize { self.capacity }
    #[inline(always)]
    fn set_capacity(&mut self, capacity: usize) { self.capacity = capacity; }
    #[inline(always)]
    fn capacity_for(other: &Mat<T>, y_hier_label: AlgorithmStep, x_hier_label: AlgorithmStep, hier: &Vec<AlgorithmStep>) -> usize {
        if other.height() <= 0 || other.width() <= 0 {
            0
        } else {
            let y_tlds = match Self::get_top_level_dim_size(&hier, y_hier_label) {
                Some(a) => a,
                None => LH::to_usize(),
            };
            let x_tlds = match Self::get_top_level_dim_size(&hier, x_hier_label) {
                Some(a) => a,
                None => LW::to_usize(),
            };

             let n_blocks_y = (other.height()-1) / y_tlds + 1;
             let n_blocks_x = (other.width()-1) / x_tlds + 1;
             let other_capacity = (n_blocks_y + 1) * y_tlds * (n_blocks_x + 1) * x_tlds;
             other_capacity
        }
    }

    //Reallocate buffer if too small
    #[inline(always)]
    fn aquire_buffer_for(&mut self, req_capacity: usize) {
        let req_padded_capacity = req_capacity;
        if req_padded_capacity > self.capacity {
            unsafe {
                heap::deallocate(self.buffer as *mut _, mem::size_of::<T>() * self.capacity, 4096);
                self.buffer = heap::allocate(req_padded_capacity * mem::size_of::<T>(), 4096) as *mut _;
                self.capacity = req_padded_capacity;
            }
        }
    }

    //(But maybe only need to change y_hierarchy[self.yh_index] and x_hierarchy[self.xh_index])
    #[inline(always)]
    fn resize_to(&mut self, other: &Mat<T>, y_hier_label: AlgorithmStep, x_hier_label: AlgorithmStep, hier: &Vec<AlgorithmStep>) {
		debug_assert!(self.y_views.len() == 1, "Can't resize a submatrix!");

        let mut y_view = self.y_views.last_mut().unwrap();
        let mut x_view = self.x_views.last_mut().unwrap();

		//Adjust the size of this matrix
        y_view.iter_size = other.iter_height();
        x_view.iter_size = other.iter_width();
        y_view.padding = other.logical_h_padding();
        x_view.padding = other.logical_w_padding();

        let y_tlds = self.y_hierarchy[self.yh_index].blksz;
        let x_tlds = self.x_hierarchy[self.xh_index].blksz;

        let n_blocks_y = (other.height()-1) / y_tlds + 1;
        let n_blocks_x = (other.width()-1) / x_tlds + 1;
        let h_padded = n_blocks_y * y_tlds;
        let w_padded = n_blocks_x * x_tlds;
		
		//Need to rebuild the hierarchy to adjust the strides, since the dimensions changed. 
        let (y_hierarchy, x_hierarchy) = Self::parse_input_hierarchy(h_padded, w_padded, &hier, y_hier_label, x_hier_label);
		self.y_hierarchy = y_hierarchy;
		self.x_hierarchy = x_hierarchy;
        self.yh_index = 0;
        self.xh_index = 0;
    }
}

impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> RoCM<T> for
    Hierarch<T, LH, LW, LRS, LCS> {
    #[inline(always)]
    fn partition_is_rocm(&self) -> bool { 
        self.height() <= LH::to_usize() && self.width() <= LW::to_usize()
    }

    #[inline(always)]
    fn get_leaf_rs(&self) -> usize { 
        LRS::to_usize()
    }

    #[inline(always)]
    fn get_leaf_cs(&self) -> usize { 
        LCS::to_usize()
    }

    #[inline(always)]
    unsafe fn get_buffer(&self) -> *const T {
        let y_off = self.y_views.last().unwrap().offset;
        let x_off = self.x_views.last().unwrap().offset;
        self.buffer.offset((y_off + x_off) as isize) 
    }

    #[inline(always)]
    unsafe fn get_mut_buffer(&mut self) -> *mut T {
        let y_off = self.y_views.last().unwrap().offset;
        let x_off = self.x_views.last().unwrap().offset;
        self.buffer.offset((y_off + x_off) as isize) 
    }

    #[inline(always)]
    fn get_block_rs(&self, lvl: usize, blksz: usize) -> usize {
        if lvl == 0 { 
            LRS::to_usize()
        } else {
            let index = self.y_hierarchy.len() - lvl - 1;
            debug_assert!(self.y_hierarchy[index].blksz == blksz);
            self.y_hierarchy[index].stride
        }
    }
    #[inline(always)]
    fn get_block_cs(&self, lvl: usize, blksz: usize) -> usize {
        if lvl == 0 { 
            LCS::to_usize()
        } else {
            let index = self.x_hierarchy.len() - lvl - 1;
            debug_assert!(self.x_hierarchy[index].blksz == blksz);
            self.x_hierarchy[index].stride
        }
    }
}
