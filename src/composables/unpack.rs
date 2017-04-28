use core::ptr::{self};
use core::marker::PhantomData;
use core::cmp;

use matrix::{Scalar,Mat,Hierarch,Matrix,ResizableBuffer,HierarchyNode,RoCM};
use typenum::Unsigned;
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};

pub trait Adder <T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    fn add(a: &mut At, a_pack: &mut Apt, thr: &ThreadInfo<T>);
}

//Adds Apt to At.
pub struct Unpacker<T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _apt: PhantomData<Apt>,
}

//Default implementation of Unpacker. Uses the getters and setters of Mat<T>
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Adder<T, At, Apt> 
    for Unpacker<T, At, Apt> {
    default fn add(a: &mut At, a_pack: &mut Apt, thr: &ThreadInfo<T>) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        let cols_per_thread = (a.width()-1) / thr.num_threads() + 1;
        let start = cols_per_thread * thr.thread_id();
        let end = cmp::min(a.width(), start+cols_per_thread);

        for x in start..end {
            for y in 0..a.height() { 
                let alpha = a_pack.get(y,x) + a.get(y,x);
                a.set(y,x,alpha);
            }
        }
    }
}

//returns the depth and score of the level with best parallelizability
fn score_parallelizability(m: usize, y_hier: &[HierarchyNode]) -> (usize, f64)  {
    let mut best_depth = 0;
    let mut best_score = 0.0;
    let mut m_tracker = m;

    for i in 0..y_hier.len() {
        let blksz = y_hier[i].blksz;
        let n_iter = (m_tracker as f64) / (blksz as f64);
        let score = n_iter;
        if score > best_score {
            best_score = score;
            best_depth = i;
        }
        m_tracker = blksz;
    }
    (best_depth, best_score)
}

fn unpack_hier_leaf<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
	(a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>,
	 x_parallelize_level: isize, x_threads: usize, x_id: usize, 
     y_parallelize_level: isize, y_threads: usize, y_id: usize) {
    //Parallelize Y direction
    let mut ystart = 0;
    let mut yend = a.height();
    if y_parallelize_level == 0 {
		let rows_per_thread = (a.height()-1) / y_threads + 1; // micro-panels per thread
		ystart = rows_per_thread*y_id;
		yend   = ystart+rows_per_thread;
    }
    //Parallelize X direction
    let mut xstart = 0;
    let mut xend = a.width();
    if x_parallelize_level == 0 {
		let rows_per_thread = (a.width()-1) / x_threads + 1; // micro-panels per thread
		xstart = rows_per_thread*x_id;
		xend   = xstart+rows_per_thread;
    }
    unsafe{
        let cs_a = a.get_column_stride();
        let rs_a = a.get_row_stride();
        let ap = a.get_mut_buffer();

        let a_pack_p = a_pack.get_mut_buffer();

        for y in ystart..yend {
            for x in xstart..xend {
                let alpha_a = ptr::read(ap.offset((y*rs_a + x*cs_a) as isize));
                let alpha_ap = ptr::read(a_pack_p.offset((y*LRS::to_usize() + x*LCS::to_usize()) as isize));
                ptr::write(ap.offset((y*rs_a + x*cs_a) as isize), alpha_a + alpha_ap);
            }
        }
    }
}
fn unpack_hier_y<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
    (a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, y_hier: &[HierarchyNode], 
    x_parallelize_level: isize, x_threads: usize, x_id: usize,
    y_parallelize_level: isize, y_threads: usize, y_id: usize) {
    if y_hier.len()-1 == 0 { 
        unpack_hier_leaf(a, a_pack, x_parallelize_level, x_threads, x_id, y_parallelize_level, y_threads, y_id);
    } else {
        let blksz = y_hier[0].blksz;
        let m_save = a.push_y_view(blksz);
        a_pack.push_y_view(blksz);
        
        let mut start = 0;
        let mut end = m_save;
        if y_parallelize_level == 0 {
			let range = m_save;
			let n_blocks = (range-1) / blksz + 1;         
			let blocks_per_thread = (n_blocks-1) / y_threads + 1; // micro-panels per thread
			start = blksz*blocks_per_thread*y_id;
			end   = start+blksz*blocks_per_thread;
        }
        let mut i = start;
        while i < end {
            a.slide_y_view_to(i, blksz);
            a_pack.slide_y_view_to(i, blksz);
            
            unpack_hier_y(a, a_pack, &y_hier[1..y_hier.len()], 
                x_parallelize_level, x_threads, x_id, 
                y_parallelize_level-1, y_threads, y_id);
            i += blksz;
        }   
        a.pop_y_view();
        a_pack.pop_y_view();
    }
}
fn unpack_hier_x<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
    (a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, x_hier: &[HierarchyNode], y_hier: &[HierarchyNode],
	 x_parallelize_level: isize, x_threads: usize, x_id: usize, 
     y_parallelize_level: isize, y_threads: usize, y_id: usize)
{
    if x_hier.len() - 1 == 0 {
        unpack_hier_y(a, a_pack, y_hier, x_parallelize_level, x_threads, x_id, y_parallelize_level, y_threads, y_id);
    } else {
        let blksz = x_hier[0].blksz;
        let n_save = a.push_x_view(blksz);
        a_pack.push_x_view(blksz);
        
        let mut start = 0;
        let mut end = n_save;
        if x_parallelize_level == 0 {
			let range = n_save;
			let n_blocks = (range-1) / blksz + 1;         
			let blocks_per_thread = (n_blocks-1) / x_threads + 1; // micro-panels per thread
			start = blksz*blocks_per_thread*x_id;
			end   = start+blksz*blocks_per_thread;
        }
        let mut j = start;
        while j < end  {
            a.slide_x_view_to(j, blksz);
            a_pack.slide_x_view_to(j, blksz);
            unpack_hier_x(a, a_pack, &x_hier[1..x_hier.len()], y_hier, 
                x_parallelize_level-1, x_threads, x_id, 
                y_parallelize_level, y_threads, y_id);
            j += blksz;
        }   
        a.pop_x_view();
        a_pack.pop_x_view();
    }
}

//Specialized implementation of Unpacker for adding to Matrix<T> from Hierarch<T>
impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
    Adder<T, Matrix<T>, Hierarch<T, LH, LW, LRS, LCS>> 
    for Unpacker<T, Matrix<T>, Hierarch<T, LH, LW, LRS, LCS>> {
    default fn add(a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, thr: &ThreadInfo<T>) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }

        //Get copies of the x and y hierarchy.
        //Since we borrow a_pack as mutable during pack_hier_x,
        //we can't borrow x_hier and y_hier immutably so we must copy
        let x_hier : Vec<HierarchyNode> = 
        {
            let tmp1 = a_pack.get_x_hierarchy();
            let mut tmp2 : Vec<HierarchyNode> = Vec::with_capacity(tmp1.len());
            tmp2.extend_from_slice(tmp1);
            tmp2
        };
        let y_hier : Vec<HierarchyNode> = 
        {
            let tmp1 = a_pack.get_y_hierarchy();
            let mut tmp2 : Vec<HierarchyNode> = Vec::with_capacity(tmp1.len());
            tmp2.extend_from_slice(tmp1);
            tmp2
        };
        let (y_depth, y_score) = score_parallelizability(a.height(), &y_hier);
        let (x_depth, x_score) = score_parallelizability(a.width(), &x_hier);

		//Figure out x and y num threads
		let mut index = f64::sqrt(thr.num_threads() as f64) as usize;
        while (thr.num_threads() % index) != 0 {
            index = index - 1;
        }
        let (y_nt, x_nt) = (4,1);
            if y_score < x_score {
                (index, thr.num_threads() / index)
            } else {
                (thr.num_threads() / index, index)
            };
        
        let x_tid = thr.thread_id() / y_nt;
        let y_tid = thr.thread_id() % y_nt;

        unpack_hier_x(a, a_pack, &x_hier, &y_hier, x_depth as isize, x_nt, x_tid, y_depth as isize, y_nt, y_tid);
    }
}

pub struct UnpackC<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, CPt: Mat<T>, 
    S: GemmNode<T, At, Bt, CPt>> {
    child: S,
    c_pack: CPt,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, CPt: Mat<T>, S: GemmNode<T, At, Bt, CPt>> UnpackC<T,At,Bt,Ct,CPt,S> 
    where CPt: ResizableBuffer<T> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, CPt: Mat<T>, S: GemmNode<T, At, Bt, CPt>>
    GemmNode<T, At, Bt, Ct> for UnpackC<T, At, Bt, Ct, CPt, S>
    where CPt: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        let capacity_for_cpt = CPt:: capacity_for(c, y_marker, x_marker, &algo_desc);

        thr.barrier();

        //Check if we need to resize packing buffer
        if self.c_pack.capacity() < capacity_for_cpt {
            if thr.thread_id() == 0 {
                self.c_pack.aquire_buffer_for(capacity_for_cpt);
            }
            else {
                self.c_pack.set_capacity(capacity_for_cpt);
            }
            self.c_pack.send_alias(thr);
        }

        //Logically resize the c_pack matrix
        self.c_pack.resize_to(c, y_marker, x_marker, &algo_desc);
        //thr.barrier();
        self.c_pack.set_scalar(T::zero());
        self.child.run(a, b, &mut self.c_pack, thr);
        thr.barrier();
        <Unpacker<T,Ct,CPt>>::add(c, &mut self.c_pack, thr);
        thr.barrier();
    }
    fn new() -> UnpackC<T, At, Bt, Ct, CPt, S>{
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};

        UnpackC{ child: S::new(), 
               c_pack: CPt::empty(y_marker, x_marker, &algo_desc),
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}
