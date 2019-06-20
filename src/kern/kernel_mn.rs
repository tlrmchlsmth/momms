use matrix::{Scalar,Mat,RoCM,Matrix};
use core::ptr;
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use super::ukernel_wrapper::{UkernelWrapper,GenericUkernelWrapper};

pub struct KernelMN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, const Mr: usize, const Nr: usize>{
    tmp: Matrix<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, const Mr: usize, const Nr: usize> 
    GemmNode<T, At, Bt, Ct> for KernelMN<T, At, Bt, Ct, {Mr}, {Nr}> 
    where At: RoCM<T>, Bt: RoCM<T>, Ct: RoCM<T> {
    #[inline(always)]
    default unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T>) -> () {
        //A must be column major and B must be row major 
        debug_assert!(a.get_leaf_rs() == 1 && a.get_leaf_cs() == Mr);
        debug_assert!(b.get_leaf_cs() == 1 && b.get_leaf_rs() == Nr);

        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha = a.get_scalar() * b.get_scalar();
        let mut beta = c.get_scalar();

        let c_leaf_rs = c.get_leaf_rs() as isize;
        let c_leaf_cs = c.get_leaf_cs() as isize;

        let c_mr_stride = c.get_block_rs(1, Mr) as isize;
        let a_mr_stride = a.get_block_rs(1, Mr) as isize;

        let c_nr_stride = c.get_block_cs(1, Nr) as isize;
        let b_nr_stride = b.get_block_cs(1, Nr) as isize;

        let mut ir : isize = 0;
        let mut a_ir = ap;
        let mut c_ir = cp;
        while ir < m {
    
            let mut c_jr = c_ir;
            let mut b_jr = bp;
            let mut jr : isize = 0;
            while jr < n {
                if Ct::full_leaves() || (n - jr >= Nr as isize) && (m - ir >= Mr as isize) {
                    <UkernelWrapper<T,{Mr},{Nr}>>::run(k, &mut alpha, a_ir, b_jr, &mut beta, c_jr, c_leaf_rs, c_leaf_cs);
                } else {
					let tp = self.tmp.get_mut_buffer();
					let mut t_scalar = T::zero();
                    let t_rs = self.tmp.get_row_stride() as isize;
                    let t_cs = self.tmp.get_column_stride() as isize;
					<UkernelWrapper<T,{Mr},{Nr}>>::run(k, &mut alpha, a_ir, b_jr, &mut t_scalar, tp, t_rs, t_cs);

                    let local_m = if m-ir >= Mr as isize { Mr as isize } else { m-ir };
                    let local_n = if n-jr >= Nr as isize { Nr as isize } else { n-jr };

					//Add t to c
                    for ii in 0..local_m {
                        for jj in 0..local_n {
                            let tau = ptr::read(tp.offset(ii * t_rs + jj * t_cs));
                            let chi = ptr::read(c_jr.offset(ii * c_leaf_rs + jj * c_leaf_cs));
                            ptr::write(c_jr.offset(ii * c_leaf_rs + jj * c_leaf_cs), tau+beta*chi);
                        }
                    }
                }

                jr += Nr as isize;
                c_jr = c_jr.offset(c_nr_stride);
                b_jr = b_jr.offset(b_nr_stride);
            }
            ir += Mr as isize;
            a_ir = a_ir.offset(a_mr_stride);
            c_ir = c_ir.offset(c_mr_stride);
        }
    }
    fn new() -> Self {
        let mut tmp = <Matrix<T>>::new(Nr, Mr);
        tmp.transpose();
        KernelMN{ tmp: tmp, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::N{bsz: Nr});
        desc.push(AlgorithmStep::M{bsz: Mr});
        desc
    }  
}
