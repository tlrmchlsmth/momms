use core::ptr::{self};
use core::marker::PhantomData;
use core::cmp;

use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Hierarch,Matrix,ResizableBuffer,HierarchyNode};
use typenum::Unsigned;
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};

//This trait exists so that Packer has a type to specialize over.
//Yes this is stupid.
pub trait Copier <T: Scalar, At: Mat<T>, Apt: Mat<T>> {
        fn pack( &self, a: &mut At, a_pack: &mut Apt, thr: &ThreadInfo<T> );
}


pub struct Packer<T: Scalar, At: Mat<T>, Apt: Mat<T>> {
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _apt: PhantomData<Apt>,
}
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Packer<T, At, Apt> {
    fn new() -> Packer<T, At, Apt> {
        Packer{ _t: PhantomData, _at: PhantomData, _apt: PhantomData } 
    }
}
    
//Default implementation of Packer. Uses the getters and setters of Mat<T>
impl<T: Scalar, At: Mat<T>, Apt: Mat<T>> Copier<T, At, Apt> 
    for Packer<T, At, Apt> {
    default fn pack( &self, a: &mut At, a_pack: &mut Apt, thr: &ThreadInfo<T> ) {
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
    fn pack( &self, a: &mut Matrix<T>, a_pack: &mut ColumnPanelMatrix<T, PW>, thr: &ThreadInfo<T> ) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        unsafe {
            let ap = a.get_buffer();
            let cs_a = a.get_column_stride();
            let rs_a = a.get_row_stride();
            
            let n_panels = (a_pack.width()-1) / PW::to_usize() + 1;
            let panels_per_thread = (n_panels-1) / thr.num_threads() + 1;
            let start = panels_per_thread * thr.thread_id();
            let end = cmp::min(n_panels, start+panels_per_thread);

            for panel in start..end-1 {
                let p = a_pack.get_panel(panel);
                let ap1 = ap.offset((panel * PW::to_usize() * cs_a) as isize);

                for y in 0..a_pack.height() {
                    for i in 0..PW::to_usize() {
                        let alpha = ptr::read(ap1.offset((y*rs_a + i*cs_a) as isize));
                        ptr::write( p.offset((y*PW::to_usize() + i) as isize), alpha );
                    }
                }
            }
            //Handle the last panel separately, since it might have fewer than panel width number of columns
            let last_panel_w = if end * PW::to_usize() > a_pack.width() {
                a_pack.width() - (end-1)*PW::to_usize()
            } else {
                PW::to_usize()
            };
            let p = a_pack.get_panel(end-1); 
            let ap1 = ap.offset(((end-1) * PW::to_usize() * cs_a) as isize);
            for y in 0..a_pack.height() {
                for i in 0..last_panel_w {
                    let alpha = ptr::read(ap1.offset((y*rs_a + i*cs_a)as isize));
                    ptr::write( p.offset((y*PW::to_usize() + i) as isize), alpha );
                }
            }
        }
    }
}

//Specialized implementation of Packer for packing from general strided matrices into row panel
//matrices
impl<T: Scalar, PH: Unsigned> Copier<T, Matrix<T>, RowPanelMatrix<T, PH>> 
    for Packer<T, Matrix<T>, RowPanelMatrix<T, PH>> {
    fn pack( &self, a: &mut Matrix<T>, a_pack: &mut RowPanelMatrix<T, PH>, thr: &ThreadInfo<T> ) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }
        unsafe {
            let ap = a.get_buffer();
            let cs_a = a.get_column_stride();
            let rs_a = a.get_row_stride();

            let n_panels = (a_pack.height()-1) / PH::to_usize() + 1;
            let panels_per_thread = (n_panels-1) / thr.num_threads() + 1;
            let start = panels_per_thread * thr.thread_id();
            let end = cmp::min(n_panels, start+panels_per_thread);

            for panel in start..end-1 {
                let p = a_pack.get_panel(panel); 
                let ap1 = ap.offset((panel * PH::to_usize() * rs_a) as isize); 

                for x in 0..a_pack.width() {
                    for i in 0..PH::to_usize() {
                        let alpha = ptr::read(ap1.offset((x*cs_a + i*rs_a) as isize));
                        ptr::write( p.offset((x*PH::to_usize() + i) as isize), alpha );
                    }
                }
            }
            //Handle the last panel separately, since it might have fewer than panel width number of columns
            let last_panel_h = if end * PH::to_usize() > a_pack.height() {
                a_pack.height() - (end-1)*PH::to_usize()
            } else {
                PH::to_usize()
            };
            let p = a_pack.get_panel(end-1); 
            let ap1 = ap.offset(((end-1) * PH::to_usize() * rs_a) as isize); 
            for x in 0..a_pack.width() {
                for i in 0..last_panel_h {
                    let alpha = ptr::read(ap1.offset((x*cs_a + i*rs_a)as isize));
                    ptr::write( p.offset((x*PH::to_usize() + i) as isize), alpha );
                }
            }
        }
    }
}


fn pack_hier_leaf<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
(a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>) {
/*    for x in 0..a.width() {
        for y in 0..a.height() { 
            a_pack.set(y, x, a.get(y,x));
        }
    }*/
    unsafe{
        let cs_a = a.get_column_stride();
        let rs_a = a.get_row_stride();
        let ap = a.get_buffer();

        let mut a_pack_p = a_pack.get_mut_buffer();

        for y in 0..a.height() {
            for x in 0..a.width() {
                let alpha = ptr::read(ap.offset((y*rs_a + x*cs_a) as isize));
                ptr::write( a_pack_p.offset((y*LRS::to_usize() + x*LCS::to_usize()) as isize), alpha );
            }
        }
    }
}
fn pack_hier_y<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
(a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, y_hier: &[HierarchyNode]){
    if y_hier.len()-1 == 0 { 
        pack_hier_leaf(a, a_pack);
    } else {
        let blksz = y_hier[0].blksz;
        let m_save = a.push_y_view(blksz);
        a_pack.push_y_view(blksz);
        
        let mut i = 0;
        while i < m_save {
            a.slide_y_view_to(i, blksz);
            a_pack.slide_y_view_to(i, blksz);
            
            pack_hier_y(a, a_pack, &y_hier[1..y_hier.len()]);
            i += blksz;
        }   
        a.pop_y_view();
        a_pack.pop_y_view();
    }
}
fn pack_hier_x<T: Scalar, LH: Unsigned, LW: Unsigned, LRS: Unsigned, LCS: Unsigned> 
(a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, x_hier: &[HierarchyNode], y_hier: &[HierarchyNode])
{
    if x_hier.len()-1 == 0 {
        pack_hier_y(a, a_pack, y_hier);
    } else {
        let blksz = x_hier[0].blksz;
        let n_save = a.push_x_view(blksz);
        a_pack.push_x_view(blksz);
        
        let mut j = 0;
        while j < n_save  {
            a.slide_x_view_to(j, blksz);
            a_pack.slide_x_view_to(j, blksz);
            pack_hier_x(a, a_pack, &x_hier[1..x_hier.len()], y_hier);
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
    default fn pack( &self, a: &mut Matrix<T>, a_pack: &mut Hierarch<T, LH, LW, LRS, LCS>, thr: &ThreadInfo<T> ) {
        if a_pack.width() <= 0 || a_pack.height() <= 0 {
            return;
        }

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
        pack_hier_x(a, a_pack, &x_hier, &y_hier);
    }
}


pub struct PackA<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, APt: Mat<T>, 
    S: GemmNode<T, APt, Bt, Ct>> {
    child: S,
    packer: Packer<T, At, APt>,
    a_pack: APt,
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
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T> ) -> () {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        let capacity_for_apt = APt:: capacity_for(a, y_marker, x_marker, &algo_desc);
        thr.barrier();
        if self.a_pack.capacity() < capacity_for_apt {
            if thr.thread_id() == 0 {
                self.a_pack.aquire_buffer_for(capacity_for_apt);
            }
            else {
                self.a_pack.set_capacity( capacity_for_apt );
            }
            self.a_pack.send_alias( thr );
        }

        self.a_pack.resize_to( a, y_marker, x_marker, &algo_desc );
        self.packer.pack( a, &mut self.a_pack, thr );
        thr.barrier();
        self.child.run(&mut self.a_pack, b, c, thr);
    }
    fn new( ) -> PackA<T, At, Bt, Ct, APt, S>{
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::M{bsz: 0};
        let x_marker = AlgorithmStep::K{bsz: 0};

        PackA{ child: S::new(), 
               a_pack: APt::empty(y_marker, x_marker, &algo_desc), packer: Packer::new(),
               _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}

pub struct PackB<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, 
    S: GemmNode<T, At, BPt, Ct>> {
    child: S,
    packer: Packer<T, Bt, BPt>,
    b_pack: BPt,
    _at: PhantomData<At>,
    _ct: PhantomData<Ct>,
} 
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, S: GemmNode<T, At, BPt, Ct>> PackB <T,At,Bt,Ct,BPt,S> 
    where BPt: ResizableBuffer<T> {
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, BPt: Mat<T>, S: GemmNode<T, At, BPt, Ct>>
    GemmNode<T, At, Bt, Ct> for PackB<T, At, Bt, Ct, BPt, S>
    where BPt: ResizableBuffer<T> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T> ) -> () {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};

        let capacity_for_bpt = BPt:: capacity_for(b, y_marker, x_marker, &algo_desc);
        thr.barrier();
        if self.b_pack.capacity() < capacity_for_bpt {
            if thr.thread_id() == 0 {
                self.b_pack.aquire_buffer_for(capacity_for_bpt);
            }
            else {
                self.b_pack.set_capacity(capacity_for_bpt);
            }
            self.b_pack.send_alias( thr );
        }
        self.b_pack.resize_to(b, y_marker, x_marker, &algo_desc);
        self.packer.pack( b, &mut self.b_pack, thr );
        thr.barrier();
        self.child.run(a, &mut self.b_pack, c, thr);
    }
    fn new( ) -> PackB<T, At, Bt, Ct, BPt, S> {
        let algo_desc = S::hierarchy_description();
        let y_marker = AlgorithmStep::K{bsz: 0};
        let x_marker = AlgorithmStep::N{bsz: 0};

        PackB{ child: S::new(), 
               b_pack: BPt::empty(y_marker, x_marker, &algo_desc), packer: Packer::new(),
               _at:PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}
