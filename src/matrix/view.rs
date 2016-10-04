use core::cmp;

//Sort of a generic matrix view when you don't need anything special
#[derive(Copy,Clone)]
pub struct MatrixView{
    pub offset: usize,
    pub padding: usize,
    pub iter_size: usize,
}
impl MatrixView {
    #[inline(always)]
    pub fn physical_size( &self ) -> usize {
        if self.padding < self.iter_size {
            self.iter_size - self.padding
        } else {
            0   
        }   
    }
    #[inline(always)]
    pub fn zoomed_size_and_padding( &self, index: usize, blksz: usize ) -> (usize, usize) {
        let zoomed_iter_size = cmp::min(blksz, self.iter_size - index);
        let unzoomed_physical_size = self.physical_size();

        let zoomed_padding = if index + zoomed_iter_size < unzoomed_physical_size {
            0
        } else {
            index + zoomed_iter_size - cmp::max(index, unzoomed_physical_size)
        };
        (zoomed_iter_size, zoomed_padding)
    }
}
