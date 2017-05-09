use core::{ptr,cmp};
use core::marker::PhantomData;

use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Hierarch,Matrix,ResizableBuffer,HierarchyNode,RoCM};
use typenum::Unsigned;
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};

//returns (a,b)
//a*b = nt such that a >= b
fn decompose(nt: usize) -> (usize, usize) {
    let mut index = f64::sqrt(nt as f64) as usize;
    while (nt % index) != 0 { index = index - 1; }
    (nt / index, index)
}


//This trait exists so that Packer has a type to specialize over.
//Yes this is stupid.
pub trait Copier <T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    fn pack(a: &mut At, a_pack: &mut Apt, thr: &ThreadInfo<T>);
}


pub struct Packer<T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _apt: PhantomData<Apt>,
}
//Default implementation of Packer. Uses the getters and setters of Mat<T>
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Copier<T, At, Apt> 
    for Packer<T, At, Apt> {
    default fn pack(a: &mut At, a_pack: &mut Apt, thr: &ThreadInfo<T>) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        let cols_per_thread = (a.width()-1) / thr.num_threads() + 1;
        let start = cols_per_thread * thr.thread_id();
        let end = cmp::min(a.width(), start+cols_per_thread);

        for x in start..end {
            for y in 0..a.height() { 
                a_pack.set(y, x, a.get(y,x));
            }
        }
    }
}
 
//Specialized implementation of Packer for packing from general strided matrices into column panel
//matrices
impl<T: Scalar, PW: Unsigned> Copier<T, Matrix<T>, ColumnPanelMatrix<T, PW>> 
    for Packer<T, Matrix<T>, ColumnPanelMatrix<T, PW>> {
    fn pack(a: &mut Matrix<T>, a_pack: &mut ColumnPanelMatrix<T, PW>, thr: &ThreadInfo<T>) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        unsafe {
            let ap = a.get_buffer();
            let cs_a = a.get_column_stride();
            let rs_a = a.get_row_stride();

            let (y_nt, x_nt) = decompose(thr.num_threads());
            let x_tid = thr.thread_id() / y_nt;
            let y_tid = thr.thread_id() % y_nt;

            //Figure out this thread's work in x direction
            let n_panels = (a_pack.width()-1) / PW::to_usize() + 1;
            let panels_per_thread = (n_panels-1) / x_nt + 1;
            let start_panel = panels_per_thread * x_tid;
            let end_panel = cmp::min(n_panels, start_panel+panels_per_thread);

            //Figure out this thread's work in y direction
            let rows_per_thread = (a_pack.height()-1) / y_nt + 1;
            let start_row = rows_per_thread * y_tid;
            let end_row = cmp::min(a_pack.height(), start_row+rows_per_thread);

            for panel in start_panel..end_panel {
                let p = a_pack.get_panel(panel);
                let ap1 = ap.offset((panel * PW::to_usize() * cs_a) as isize);

                for y in start_row..end_row {
                    for i in 0..PW::to_usize() {
                        let alpha = ptr::read(ap1.offset((y*rs_a + i*cs_a) as isize));
                        ptr::write(p.offset((y*PW::to_usize() + i) as isize), alpha);
                    }
                }
            }
        }
    }
}

//Specialized implementation of Packer for packing from general strided matrices into row panel
//matrices
impl<T: Scalar, PH: Unsigned> Copier<T, Matrix<T>, RowPanelMatrix<T, PH>> 
    for Packer<T, Matrix<T>, RowPanelMatrix<T, PH>> {
    fn pack(a: &mut Matrix<T>, a_pack: &mut RowPanelMatrix<T, PH>, thr: &ThreadInfo<T>) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        unsafe {
            let ap = a.get_buffer();
            let cs_a = a.get_column_stride();
            let rs_a = a.get_row_stride();

            let (y_nt, x_nt) = decompose(thr.num_threads());
            let x_tid = thr.thread_id() / y_nt;
            let y_tid = thr.thread_id() % y_nt;

            //Figure out this thread's work in y direction
            let n_panels = (a_pack.height()-1) / PH::to_usize() + 1;
            let panels_per_thread = (n_panels-1) / y_nt + 1;
            let start_panel = panels_per_thread * y_tid;
            let end_panel = cmp::min(n_panels, start_panel+panels_per_thread);

            //Figure out this thread's work in x direction
            let cols_per_thread = (a_pack.width()-1) / x_nt + 1;
            let start_col = cols_per_thread * x_tid;
            let end_col = cmp::min(a_pack.width(), start_col+cols_per_thread);

            for panel in start_panel..end_panel {
                let p = a_pack.get_panel(panel); 
                let ap1 = ap.offset((panel * PH::to_usize() * rs_a) as isize); 

                for x in start_col..end_col {
                    for i in 0..PH::to_usize() {
                        let alpha = ptr::read(ap1.offset((x*cs_a + i*rs_a) as isize));
                        ptr::write(p.offset((x*PH::to_usize() + i) as isize), alpha);
                    }
                }
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

fn pack_hier_leaf<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
	(a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>,
	 x_parallelize_level: isize, x_threads: usize, x_id: usize, 
     y_parallelize_level: isize, y_threads: usize, y_id: usize) {
    //Parallelize Y direction
    let mut ystart = 0;
    let mut yend = a.height();
    if y_parallelize_level == 0 {
		let rows_per_thread = (a.height()-1) / y_threads + 1; // micro-panels per thread
		ystart = rows_per_thread*y_id;
		yend   = cmp::min(a.height(), ystart+rows_per_thread);
    }
    //Parallelize X direction
    let mut xstart = 0;
    let mut xend = a.width();
    if x_parallelize_level == 0 {
		let cols_per_thread = (a.width()-1) / x_threads + 1; // micro-panels per thread
		xstart = cols_per_thread*x_id;
		xend   = cmp::min(a.width(), xstart+cols_per_thread);
    }
    unsafe{
        let cs_a = a.get_column_stride();
        let rs_a = a.get_row_stride();
        let ap = a.get_buffer();

        let a_pack_p = a_pack.get_mut_buffer();

        for y in ystart..yend {
            for x in xstart..xend {
                let alpha = ptr::read(ap.offset((y*rs_a + x*cs_a) as isize));
                ptr::write(a_pack_p.offset((y*LRS::to_usize() + x*LCS::to_usize()) as isize), alpha);
            }
        }
    }
}
fn pack_hier_y<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
    (a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, y_hier: &[HierarchyNode], 
    x_parallelize_level: isize, x_threads: usize, x_id: usize,
    y_parallelize_level: isize, y_threads: usize, y_id: usize) {
    if y_hier.len()-1 == 0 { 
        pack_hier_leaf(a, a_pack, x_parallelize_level, x_threads, x_id, y_parallelize_level, y_threads, y_id);
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
			end   = cmp::min(m_save, start+blksz*blocks_per_thread);
        }
        let mut i = start;
        while i < end {
            a.slide_y_view_to(i, blksz);
            a_pack.slide_y_view_to(i, blksz);
            
            pack_hier_y(a, a_pack, &y_hier[1..y_hier.len()], 
                x_parallelize_level, x_threads, x_id, 
                y_parallelize_level-1, y_threads, y_id);
            i += blksz;
        }   
        a.pop_y_view();
        a_pack.pop_y_view();
    }
}
fn pack_hier_x<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
    (a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, x_hier: &[HierarchyNode], y_hier: &[HierarchyNode],
	 x_parallelize_level: isize, x_threads: usize, x_id: usize, 
     y_parallelize_level: isize, y_threads: usize, y_id: usize)
{
    if x_hier.len() - 1 == 0 {
        pack_hier_y(a, a_pack, y_hier, x_parallelize_level, x_threads, x_id, y_parallelize_level, y_threads, y_id);
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
			end   = cmp::min(n_save, start+blksz*blocks_per_thread);
        }
        let mut j = start;
        while j < end  {
            a.slide_x_view_to(j, blksz);
            a_pack.slide_x_view_to(j, blksz);
            pack_hier_x(a, a_pack, &x_hier[1..x_hier.len()], y_hier, 
                x_parallelize_level-1, x_threads, x_id, 
                y_parallelize_level, y_threads, y_id);
            j += blksz;
        }   
        a.pop_x_view();
        a_pack.pop_x_view();
    }
}

//Specialized implementation of Packer for packing from Matrix<T> into Hierarch<T>
impl<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
    Copier<T, Matrix<T>, Hierarch<T, LH, LW, LRS, LCS>> 
    for Packer<T, Matrix<T>, Hierarch<T, LH, LW, LRS, LCS>> {
    default fn pack(a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, thr: &ThreadInfo<T>) {
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
        let (rnt1, rnt2) = decompose(thr.num_threads());
        let (y_nt, x_nt) = if y_score > x_score {(rnt1,rnt2)} else {(rnt2,rnt1)};
        
        let x_tid = thr.thread_id() / y_nt;
        let y_tid = thr.thread_id() % y_nt;

        pack_hier_x(a, a_pack, &x_hier, &y_hier, x_depth as isize, x_nt, x_tid, y_depth as isize, y_nt, y_tid);
    }
}

pub struct PackA<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, APt: Mat<T>, 
    S: GemmNode<T, APt, Bt, Ct>> {
    child: S,
    a_pack: APt,
    algo_desc: Vec<AlgorithmStep>, 
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, APt: Mat<T>, S: GemmNode<T, APt, Bt, Ct>> PackA <T,At,Bt,Ct,APt,S> 
    where APt: ResizableBuffer<T> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, APt: Mat<T>, S: GemmNode<T, APt, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackA<T, At, Bt, Ct, APt, S>
    where APt: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        let capacity_for_apt = APt:: capacity_for(a, y_marker, x_marker, &self.algo_desc);
        thr.barrier();

        //Check if we need to resize packing buffer
        if self.a_pack.capacity() < capacity_for_apt {
            if thr.thread_id() == 0 {
                self.a_pack.aquire_buffer_for(capacity_for_apt);
            }
            else {
                self.a_pack.set_capacity(capacity_for_apt);
            }
            self.a_pack.send_alias(thr);
        }

        //Logically resize the a_pack matrix
        self.a_pack.resize_to(a, y_marker, x_marker, &self.algo_desc);
        <Packer<T, At, APt>>::pack(a, &mut self.a_pack, thr);
        thr.barrier();
        self.child.run(&mut self.a_pack, b, c, thr);
    }
    fn new() -> PackA<T, At, Bt, Ct, APt, S>{
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        PackA{ child: S::new(), 
               a_pack: APt::empty(y_marker, x_marker, &algo_desc), algo_desc: algo_desc,
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}

pub struct PackB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, 
    S: GemmNode<T, At, BPt, Ct>> {
    child: S,
    b_pack: BPt,
    algo_desc: Vec<AlgorithmStep>, 
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, S: GemmNode<T, At, BPt, Ct>> PackB <T,At,Bt,Ct,BPt,S> 
    where BPt: ResizableBuffer<T> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, S: GemmNode<T, At, BPt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackB<T, At, Bt, Ct, BPt, S>
    where BPt: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};
        let capacity_for_bpt = BPt:: capacity_for(b, y_marker, x_marker, &self.algo_desc);

        thr.barrier();

        //Check if we need to resize packing buffer
        if self.b_pack.capacity() < capacity_for_bpt {
            if thr.thread_id() == 0 {
                self.b_pack.aquire_buffer_for(capacity_for_bpt);
            }
            else {
                self.b_pack.set_capacity(capacity_for_bpt);
            }
            self.b_pack.send_alias(thr);
        }

        //Logically resize the c_pack matrix
        self.b_pack.resize_to(b, y_marker, x_marker, &self.algo_desc);
        <Packer<T, Bt, BPt>>::pack(b, &mut self.b_pack, thr);
        thr.barrier();
        self.child.run(a, &mut self.b_pack, c, thr);
    }
    fn new() -> PackB<T, At, Bt, Ct, BPt, S> {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};

        PackB{ child: S::new(), 
               b_pack: BPt::empty(y_marker, x_marker, &algo_desc), algo_desc: algo_desc,
               _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}
