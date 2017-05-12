use thread_comm::ThreadInfo;
use self::alloc::heap;
use matrix::{Scalar,Mat,RoCM};
use super::view::{MatrixView};
use core::{mem,ptr};

pub struct PackPair<T: Scalar, AT: Mat<T>, ApT: Mat<T>> {
    a: &mut AT,
    ap: &mut ApT,
}

impl<T: Scalar> Mat<T> for Matrix<T> {
    #[inline(always)]
    fn get(&self, y: usize, x: usize) -> T { self.a.get(y,x) }
    #[inline(always)]
    fn set(&mut self, y: usize, x: usize, alpha: T) { self.a.set(y,x,alpha); }
    #[inline(always)]
    fn off_y(&self) -> usize { 
        self.a.off_y()
    }
    #[inline(always)]
    fn off_x(&self) -> usize {
        self.a.off_x()
    }

    //Still need these for parallel range but it should go away
    #[inline(always)]
    fn set_off_y(&mut self, off_y: usize) { 
        self.a.set_off_y(off_y);
        self.ap.set_off_y(off_y);
    }
    #[inline(always)]
    fn set_off_x(&mut self, off_x: usize) {
        self.a.set_off_x(off_x);
        self.ap.set_off_x(off_x);
    }
    #[inline(always)]
    fn add_off_y(&mut self, start: usize) { 
        self.a.add_off_y(start);
        self.ap.add_off_y(start);
    }   
    #[inline(always)]
    fn add_off_x(&mut self, start: usize) { 
        self.a.add_off_x(start);
        self.ap.add_off_x(start);
    } 

    #[inline(always)]
    fn iter_height(&self) -> usize {
        self.a.iter_height()
    }
    #[inline(always)]
    fn iter_width(&self) -> usize { 
        self.a.iter_width()
    }
    #[inline(always)]
    fn set_iter_height(&mut self, iter_h: usize) {
        self.a.set_iter_height(iter_h);
        self.ap.set_iter_height(iter_h);
    }
    #[inline(always)]
    fn set_iter_width(&mut self, iter_w: usize) {
        self.a.set_iter_width(iter_w);
        self.ap.set_iter_width(iter_w);
    }

    #[inline(always)]
    fn logical_h_padding(&self) -> usize { 
        self.a.logical_h_padding()
    }
    #[inline(always)]
    fn logical_w_padding(&self) -> usize { 
        self.a.logical_w_padding()
    }
    #[inline(always)]
    fn set_logical_h_padding(&mut self, h_pad: usize) { 
        self.a.set_logical_h_padding(h_pad);
        self.ap.set_logical_h_padding(h_pad);
    }
    #[inline(always)]
    fn set_logical_w_padding(&mut self, w_pad: usize) {
        self.a.set_logical_w_padding(w_pad);
        self.ap.set_logical_w_padding(w_pad);
    }
    #[inline(always)]
    fn set_scalar(&mut self, alpha: T) {
        self.a.set_scalar(alpha);
        self.ap.set_scalar(alpha);
    }
    #[inline(always)]
    fn get_scalar(&self) -> T {
        self.a.get_scalar()
    }

    #[inline(always)]
    fn push_y_view(&mut self, blksz: usize) -> usize {
        self.a.push_y_view(blksz);
        self.ap.push_y_view(blksz)
    }

    #[inline(always)]
    fn push_x_view(&mut self, blksz: usize) -> usize {
        self.a.push_x_view(blksz);
        self.ap.push_x_view(blksz)
    }

    #[inline(always)]
    fn pop_y_view(&mut self) {
        self.a.y_views.pop();
        self.ap.y_views.pop();
    }
    
    #[inline(always)]
    fn pop_x_view(&mut self) {
        self.a.x_views.pop();
        self.ap.x_views.pop();
    }

    #[inline(always)]
    fn slide_y_view_to(&mut self, y: usize, blksz: usize) {
        self.a.slide_y_view_to(y, blksz);
        self.ap.slide_y_view_to(y, blksz);
    }
    
    #[inline(always)]
    fn slide_x_view_to(&mut self, x: usize, blksz: usize) {
        self.a.slide_x_view_to(x, blksz);
        self.ap.slide_x_view_to(x, blksz);
    }
    

   
    #[inline(always)]
    unsafe fn make_alias(&self) -> Self {
        let a_alias = self.a.make_alias();
        let ap_alias = self.ap.make_alias();
        PackPair{ a_alias, ap_alias };
    }

    #[inline(always)]
    unsafe fn send_alias(&mut self, thr: &ThreadInfo<T>) {
        self.a.send_alias(thr);
        self.ap.send_alias(thr);
    }
}
impl<T:Scalar> Drop for Matrix<T> {
    fn drop(&mut self) {
        self.a.drop();
        self.ap.drop();
    }
}
unsafe impl<T:Scalar> Send for Matrix<T> {}


impl<T: Scalar> RoCM<T> for Matrix<T> {
    #[inline(always)]
    fn partition_is_rocm(&self) -> bool { 
        a.partition_is_rocm() && ap.partition_is_rcom()
    }

    #[inline(always)]
    fn get_leaf_rs(&self) -> usize { 
        self.ap.get_leaf_rs()
    }

    #[inline(always)]
    fn get_leaf_cs(&self) -> usize { 
        self.ap.get_leaf_cs()
    }
    
    #[inline(always)]
    unsafe fn get_buffer(&self) -> *const T {
        self.ap.get_buffer()
    }   

    #[inline(always)]
    unsafe fn establish_leaf(&mut self, height: usize, width: usize)
    {
        let a_buf = self.a.get_buffer();
        let ap_buf = self.ap.get_mut_buffer();

        for ii in 0..height {
            for jj in 0..width {
                let alpha = ptr::read(a_buf.offset(ii * self.a.get_leaf_rs() + jj * self.a.get_leaf_cs()));
                ptr::write(ap_buf.offset(ii * self.ap.get_leaf_rs() + jj * self.ap.get_leaf_cs()), alpha);
            }
        }
    }

    #[inline(always)]
    unsafe fn get_mut_buffer(&mut self) -> *mut T {
        let y_view = self.y_views.last().unwrap();
        let x_view = self.x_views.last().unwrap();

        self.buffer.offset((y_view.offset*self.row_stride + x_view.offset*self.column_stride) as isize)
    }
    #[inline(always)]
    fn get_block_rs(&self, lvl: usize, blksz: usize) -> usize {
        ap.get_block_rs(lvl, blksz)
    }
    #[inline(always)]
    fn get_block_cs(&self, _: usize, blksz: usize) -> usize {
        ap.get_block_cs(lvl, blksz)
    }
    #[inline(always)]
    fn full_leaves() -> bool {
        ap.full_leaves()
    }
}
