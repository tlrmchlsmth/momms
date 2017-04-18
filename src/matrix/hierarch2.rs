extern crate alloc;
extern crate generic_array;

use thread_comm::ThreadInfo;
use typenum::Unsigned;
use self::alloc::heap;
use matrix::{Scalar,Mat,ResizableBuffer};
use self::generic_array::{GenericArray,ArrayLength};

use core::marker::PhantomData;
use core::mem;
use core::ptr::{self};


//I know how to push.
//But I don't know how to pop...


//The problem with this is that the partition code must have the same matrix time going in as
//coming out
//So this doesn't work...
//We can have a vector of boxes of hierarchy nodes.. 
//It's still dynamic dispatch
pub trait HierarchDescNode where Self: Mat
{
    is_alias: bool,
}

pub struct HierarchyX<T: Scalar, Bsz: Unsigned, S: HierarchNode>  {
    pub stride: usize, //This is the stride between submatrices
    offset: usize
    pub child: S,
}
pub struct HierarchyY<T: Scalar, Bsz: Unsigned, S: HierarchNode>  {
    pub stride: usize, //This is the stride between submatrices
    offset: usize
    pub child: S,
}
pub struct HierarchyLeaf<T: Scalar, M: Unsigned, N: Unsigned, RS: Unsigned, CS: Unsigned>  {
    buffer: *mut T,
    offset: usize
}



#[derive(Copy,Clone)]
pub struct HierarchyMatrixView{
    pub offset: usize,
    pub padding: usize,
    pub iter_size: usize,
    pub node: HierarchyNode,
}

//LRS is leaf row stride, LCS is leaf column stride
pub struct Hierarch<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned>{ 
    //We pop from the hierarchy nodes, and push into the view nodes (with some details added)

    //These are the view stacks just like the other Matrices
    y_views: Vec<HierarchyMatrixView>,
    x_views: Vec<HierarchyMatrixView>,

    //These describe the hierarchy of the current view
    y_hierarchy: Vec<HierarchyNode>,
    x_hierarchy: Vec<HierarchyNode>,

    _lh_t:  PhantomData<LH>,
    _lw_t:  PhantomData<LW>,
    _lrs_t: PhantomData<LRS>,
    _lcs_t: PhantomData<LCS>,
}

impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> Hierarch<T, LH, LW, LRS, LCS> {
    //Create a matrix.
    //Not sure if it makes sense to be able to create a matrix first and then describe its hierarchy
    //BUT I really like the ability to add to the hierarchy one piece at a time...
    pub fn new( h: usize, w: usize ) -> Hierarch<T,LRS,LCS> {
    }
    //Build up the hierarchy of the matrix
    pub fn add_to_y_hierarchy( &mut self ) {
    }
    pub fn add_to_x_hierarchy( &mut self ) {
    }

    //Helper functions to drill down the cache hierarchy to get the physical location of a matrix
    //element
    pub fn get_offset_x(&self, usize x) -> usize {
    }
    pub fn get_offset_y(&self, usize y) -> usize {
    }
}
impl<T: Scalar, LRS: Unsigned, LCS: Unsigned> Mat<T> for Hierarch<T, LRS, LCS> {
    
}
