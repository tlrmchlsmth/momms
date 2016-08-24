use core::marker::{PhantomData};
use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::GemmNode;

#[derive(Copy,Clone)]
pub enum ThreadsTarget {
    Target(usize),
    TheRest,
}
struct ParallelInfo<T: Scalar> {
    thr: ThreadInfo<T>,
    n_way: usize,
    work_id: usize,
}

pub struct ParallelM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    //Initialized Stuff
    child: S,
    n_threads: ThreadsTarget,
    iota: usize,

    //Info about how to parallelize, decided at runtime
    par_inf: Option<ParallelInfo<T>>,

    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    ParallelM<T,At,Bt,Ct,S> {
    pub fn new(n_threads: ThreadsTarget, iota: usize, child: S) -> ParallelM<T, At, Bt, Ct, S>{
        ParallelM{ n_threads: n_threads, child: child, iota: iota, par_inf: Option::None,
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    #[inline(always)]
    fn make_subinfo(&mut self, info: &ThreadInfo<T>) -> ParallelInfo<T>{
        //First figure out how many ways to split into
        let n_way = match self.n_threads {
            ThreadsTarget::Target(thr_target) => 
                if info.num_threads() % thr_target == 0 { thr_target } else { 1 },
            ThreadsTarget::TheRest => info.num_threads(),
        };
        let subcomm_n_threads = info.num_threads() / n_way;

        //Figure out new thread IDs
        let subinfo = info.split(n_way);
        ParallelInfo{ thr: subinfo, n_way: n_way, 
            work_id: info.thread_id() / subcomm_n_threads }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for ParallelM<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        //Split the thread communicator and create new thread infos
        let parallel_info = match self.par_inf {
            Some(ref x) => { x },
            None => { let new_par_inf = self.make_subinfo( thr );
                      self.par_inf = Option::Some( new_par_inf );
                      self.par_inf.as_ref().unwrap()
            }, 
        };

        
        //Now figure out the range of this thread
        let range = a.iter_height();                     // Global range
        let n_iotas = (range-1) / self.iota + 1;         // Number of MR micro-panels
        let iotas_per_thread = (n_iotas-1) / parallel_info.n_way + 1; // micro-panels per thread
        let start = self.iota*iotas_per_thread*parallel_info.work_id;
        let end   = start+self.iota*iotas_per_thread;

        //Partition matrices and adjust logical padding
        let h_iter_save = a.iter_height();
        let h_padding_save = a.logical_h_padding();
        let ay_off_save = a.off_y();
        let cy_off_save = c.off_y();
        let new_padding = if end <= a.height() { 0 } else { end - a.height() };
        a.set_logical_h_padding(new_padding);
        a.set_iter_height(end-start);
        a.set_off_y(ay_off_save+start);
        c.set_logical_h_padding(new_padding);
        c.set_iter_height(end-start);
        c.set_off_y(cy_off_save+start);
        
        //Run subproblem
        self.child.run(a, b, c, &parallel_info.thr);

        //Unpartition matrices
        a.set_logical_h_padding(h_padding_save);
        a.set_iter_height(h_iter_save);
        a.set_off_y(ay_off_save);
        c.set_logical_h_padding(h_padding_save);
        c.set_iter_height(h_iter_save);
        c.set_off_y(cy_off_save);
    }
    #[inline(always)]
    unsafe fn shadow(&self) -> Self where Self: Sized {
        ParallelM{ n_threads: self.n_threads, child: self.child.shadow(), iota: self.iota,
            par_inf: Option::None,
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}

pub struct ParallelN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    //Initialized Stuff
    child: S,
    n_threads: ThreadsTarget,
    iota: usize,

    //Info about how to parallelize, decided at runtime
    par_inf: Option<ParallelInfo<T>>,

    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    ParallelN<T,At,Bt,Ct,S> {
    pub fn new(n_threads: ThreadsTarget, iota: usize, child: S) -> ParallelN<T, At, Bt, Ct, S>{
        ParallelN{ n_threads: n_threads, child: child, iota: iota, par_inf: Option::None,
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    #[inline(always)]
    fn make_subinfo(&mut self, info: &ThreadInfo<T>) -> ParallelInfo<T>{
        //First figure out how many ways to split into
        let n_way = match self.n_threads {
            ThreadsTarget::Target(thr_target) => 
                if info.num_threads() % thr_target == 0 { thr_target } else { 1 },
            ThreadsTarget::TheRest => info.num_threads(),
        };
        let subcomm_n_threads = info.num_threads() / n_way;

        //Figure out new thread IDs
        let subinfo = info.split(n_way);
        ParallelInfo{ thr: subinfo, n_way: n_way, 
            work_id: info.thread_id() / subcomm_n_threads }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for ParallelN<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        //Split the thread communicator and create new thread infos
        let parallel_info = match self.par_inf {
            Some(ref x) => { x },
            None => { let new_par_inf = self.make_subinfo( thr );
                      self.par_inf = Option::Some( new_par_inf );
                      self.par_inf.as_ref().unwrap()
            }, 
        };

        
        //Now figure out the range of this thread
        let range = b.iter_width();                     // Global range
        let n_iotas = (range-1) / self.iota + 1;         // Number of MR micro-panels
        let iotas_per_thread = (n_iotas-1) / parallel_info.n_way + 1; // micro-panels per thread
        let start = self.iota*iotas_per_thread*parallel_info.work_id;
        let end   = start+self.iota*iotas_per_thread;

        //Partition matrices and adjust logical padding
        let w_iter_save = b.iter_width();
        let w_padding_save = a.logical_w_padding();
        let bx_off_save = a.off_x();
        let cx_off_save = c.off_x();
        let new_padding = if end <= b.width() { 0 } else { end - b.width() };
        b.set_logical_w_padding(new_padding);
        b.set_iter_width(end-start);
        b.set_off_x(bx_off_save+start);
        c.set_logical_w_padding(new_padding);
        c.set_iter_width(end-start);
        c.set_off_x(cx_off_save+start);


        //Run subproblem
        self.child.run(a, b, c, &parallel_info.thr);

        //Unpartition matrices
        b.set_logical_w_padding(w_padding_save);
        b.set_iter_width(w_iter_save);
        b.set_off_x(bx_off_save);
        c.set_logical_w_padding(w_padding_save);
        c.set_iter_width(w_iter_save);
        c.set_off_x(cx_off_save);
    }
    #[inline(always)]
    unsafe fn shadow(&self) -> Self where Self: Sized {
        ParallelN{ n_threads: self.n_threads, child: self.child.shadow(), iota: self.iota,
            par_inf: Option::None,
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
