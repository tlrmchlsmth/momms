use core::marker::{PhantomData};
use matrix::{Scalar,Mat};
use thread_comm::ThreadInfo;
use composables::{GemmNode,AlgorithmStep};
use typenum::Unsigned;

//Some helper types so we can specify how the parallelizers decide how many threads to use
pub trait Nwayer{
    fn get_n_way(usize) -> usize;
}
pub struct Target<Nthr: Unsigned> { _nthr: PhantomData<Nthr> }
impl<Nthr: Unsigned> Nwayer for Target<Nthr> {
    fn get_n_way(n_threads: usize) -> usize {
        if n_threads % Nthr::to_usize() == 0 { Nthr::to_usize() } else { 1 }
    }
}
pub struct TheRest { }
impl Nwayer for TheRest {
    fn get_n_way(n_threads: usize) -> usize {
        n_threads
    }
}

//Info for one thread to parallelize
struct ParallelInfo<T: Scalar> {
    thr: ThreadInfo<T>,
    n_way: usize,
    work_id: usize,
}

pub struct ParallelM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, 
    Iota: Unsigned, Nthr: Nwayer, S: GemmNode<T, At, Bt, Ct>> {
    //Initialized Stuff
    child: S,

    //Info about how to parallelize, decided at runtime
    par_inf: Option<ParallelInfo<T>>,

    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _iotat: PhantomData<Iota>,
    _nthr: PhantomData<Nthr>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, 
    Iota: Unsigned, Nthr: Nwayer, S: GemmNode<T, At, Bt, Ct>> ParallelM<T,At,Bt,Ct,Iota,Nthr,S> {
    #[inline(always)]
    fn make_subinfo(&mut self, info: &ThreadInfo<T>) -> ParallelInfo<T>{
        //First figure out how many ways to split into
        let n_way = Nthr::get_n_way(info.num_threads());
        let subcomm_n_threads = info.num_threads() / n_way;

        //Figure out new thread IDs
        let subinfo = info.split(n_way);
        ParallelInfo{ thr: subinfo, n_way: n_way, 
            work_id: info.thread_id() / subcomm_n_threads }
    }
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, 
    Iota: Unsigned, Nthr: Nwayer, S: GemmNode<T, At, Bt, Ct>> GemmNode<T, At, Bt, Ct> for ParallelM<T,At,Bt,Ct,Iota,Nthr,S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        //Split the thread communicator and create new thread infos
        let parallel_info = match self.par_inf {
            Some(ref x) => { x },
            None => { let new_par_inf = self.make_subinfo(thr);
                      self.par_inf = Option::Some(new_par_inf);
                      self.par_inf.as_ref().unwrap()
            }, 
        };

        //Determine work range of this thread
        let n_iotas = (a.iter_height() - 1) / Iota::to_usize() + 1;
        let iotas_per_thread = (n_iotas - 1) / parallel_info.n_way + 1;
        let start = Iota::to_usize()*iotas_per_thread*parallel_info.work_id;
        let end   = start+Iota::to_usize()*iotas_per_thread;

        a.push_y_split(start, end);
        c.push_y_split(start, end);

        //Run subproblem
        self.child.run(a, b, c, &parallel_info.thr);

        a.pop_y_split();
        c.pop_y_split();
    }
    fn new() -> Self {
        ParallelM{ child: S::new(), par_inf: Option::None,
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData,
            _iotat: PhantomData, _nthr: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    } 
}

pub struct ParallelN<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, 
    Iota: Unsigned, Nthr: Nwayer, S: GemmNode<T, At, Bt, Ct>> {
    //Initialized Stuff
    child: S,

    //Info about how to parallelize, decided at runtime
    par_inf: Option<ParallelInfo<T>>,

    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _iotat: PhantomData<Iota>,
    _nthr: PhantomData<Nthr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Iota: Unsigned, Nthr: Nwayer, S: GemmNode<T, At, Bt, Ct>> 
    ParallelN<T,At,Bt,Ct,Iota,Nthr,S> {
    #[inline(always)]
    fn make_subinfo(&mut self, info: &ThreadInfo<T>) -> ParallelInfo<T>{
        //First figure out how many ways to split into
        let n_way = Nthr::get_n_way(info.num_threads());
        let subcomm_n_threads = info.num_threads() / n_way;

        //Figure out new thread IDs
        let subinfo = info.split(n_way);
        ParallelInfo{ thr: subinfo, n_way: n_way, 
            work_id: info.thread_id() / subcomm_n_threads }
    }
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, 
    Iota: Unsigned, Nthr: Nwayer, S: GemmNode<T, At, Bt, Ct>> GemmNode<T, At, Bt, Ct> for ParallelN<T,At,Bt,Ct,Iota,Nthr,S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        //Split the thread communicator and create new thread infos
        let parallel_info = match self.par_inf {
            Some(ref x) => { x },
            None => { let new_par_inf = self.make_subinfo(thr);
                      self.par_inf = Option::Some(new_par_inf);
                      self.par_inf.as_ref().unwrap()
            }, 
        };

        //Determine work range of this thread
        let n_iotas = (b.iter_width() - 1) / Iota::to_usize() + 1;
        let iotas_per_thread = (n_iotas - 1) / parallel_info.n_way + 1;
        let start = Iota::to_usize()*iotas_per_thread*parallel_info.work_id;
        let end   = start+Iota::to_usize()*iotas_per_thread;

        b.push_x_split(start, end);
        c.push_x_split(start, end);

        //Run subproblem
        self.child.run(a, b, c, &parallel_info.thr);

        b.pop_x_split();
        c.pop_x_split();
    }
    fn new() -> Self {
        ParallelN{ child: S::new(), par_inf: Option::None,
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData,
            _iotat: PhantomData, _nthr: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}
