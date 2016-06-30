extern crate alloc;
use std::ptr::{self};
use std::mem;
use self::alloc::heap;
use std::sync::{Arc,Barrier,RwLock};
use std::sync::atomic::{AtomicPtr,AtomicUsize,AtomicBool,Ordering};

use matrix::{Mat,Scalar};
use core::marker::{PhantomData};
pub use gemm::{GemmNode};

extern crate crossbeam;
use self::crossbeam::{Scope};

pub struct ThreadComm<T> {
    n_threads: usize,

    //Slot has a MatrixBuffer, to be broadcast
    slot: AtomicPtr<T>,

    //Slot_reads represents the number of times slot has been read.
    //If slot_reads == n_threads, then it is ready to be written to.
    //If slot_reads < n_threads, it is ready to be read.
    //Each thread is only allowed to read from the slot one time.
    //It is incremented every time slot is read,
    //And it is an integer modulo n_threads
    slot_reads: AtomicUsize, 

    //barrier: Barrier,
    //Stuff for barriers
    barrier_sense: AtomicBool,
    barrier_threads_arrived: AtomicUsize,


    //I guess subcomms needs to have interor mutability?
    //sub_comms: Vec<AtomicPtr<Arc<ThreadComm<T>>>>,
    sub_comms: Vec<RwLock<Option<Arc<ThreadComm<T>>>>>,
}
impl<T> ThreadComm<T> {
    fn new(n_threads: usize) -> ThreadComm<T> { 
        let init_ptr: *const T = ptr::null();

        let mut sub_comms = Vec::with_capacity(n_threads);
        for i in 0..n_threads {
            sub_comms.push(RwLock::new(Option::None));
        }
        
        ThreadComm{ n_threads: n_threads,
            slot: AtomicPtr::new(init_ptr as *mut T),
            slot_reads: AtomicUsize::new(n_threads),
//            barrier: Barrier::new(n_threads),
            barrier_sense: AtomicBool::new(false),
            barrier_threads_arrived: AtomicUsize::new(0),
            sub_comms: sub_comms,
        }
    }

    fn barrier(&self, _thread_id: usize) {
        if self.n_threads == 1 {
             return;
        }

        let my_sense = self.barrier_sense.load(Ordering::Relaxed);
        let my_threads_arrived = self.barrier_threads_arrived.fetch_add(1,Ordering::Relaxed);

        if my_threads_arrived == self.n_threads-1 {   
            self.barrier_threads_arrived.store(0,Ordering::Relaxed);
            self.barrier_sense.fetch_xor(true, Ordering::Relaxed);
        } else {   
            while self.barrier_sense.load(Ordering::Relaxed) == my_sense {}
        }   

        //self.barrier.wait();
    }

    fn broadcast(&self, info: &ThreadInfo<T>, to_send: *mut T) -> *mut T {
        if info.thread_id == 0 {
            //Spin while waiting for the thread communicator to be ready to broadcast
            while self.slot_reads.load(Ordering::Relaxed) != self.n_threads {}
            self.slot.store(to_send, Ordering::Relaxed);
            self.slot_reads.store(0, Ordering::Relaxed); 
        }
        //Spin while waiting for the thread communicator chief to broadcast
        while self.slot_reads.load(Ordering::Relaxed) == self.n_threads {}
        self.slot_reads.fetch_add(1, Ordering::Relaxed);
        self.slot.load(Ordering::Relaxed)
    }
    //Pretty sure with this implementation, split can only be called one time.
    fn split(&self, thread_id: usize, n_way: usize) -> Arc<ThreadComm<T>> {
        assert!(self.n_threads % n_way == 0);

        let subcomm_n_threads = self.n_threads / n_way;
        let sub_comm_number = thread_id / subcomm_n_threads; // Which subcomm are we going to use?
        let sub_comm_id = thread_id % subcomm_n_threads; // What is our id within the subcomm?

        self.barrier(thread_id);
        if sub_comm_id == 0 {
            let mut sub_comm = self.sub_comms[sub_comm_number].write().unwrap();
            *sub_comm = Option::Some(Arc::new(ThreadComm::new(subcomm_n_threads)));
        }
        self.barrier(thread_id);

        let comm = self.sub_comms[sub_comm_number].read().unwrap().clone();
        match comm {
                Option::Some(a) => a,
                _ => unreachable!(),
        }
    }
}
//unsafe impl Sync for ThreadComm {}
//unsafe impl Send for ThreadComm {}

pub struct ThreadInfo<T> {
    thread_id: usize,
    comm: Arc<ThreadComm<T>>,
}
impl<T> ThreadInfo<T> {
    pub fn single_thread() -> ThreadInfo<T>{
        ThreadInfo{ thread_id : 0, comm : Arc::new(ThreadComm::new(1)) }
    }
    pub fn new(thread_id: usize, comm: Arc<ThreadComm<T>>) -> ThreadInfo<T> {
        ThreadInfo{ thread_id : thread_id, comm : comm.clone() }
    }
    pub fn barrier(&self) {
        self.comm.barrier(self.thread_id);
    }
    pub fn broadcast(&self, to_send: *mut T) -> *mut T {
        self.comm.broadcast(&self, to_send)
    }
    pub fn num_threads(&self) -> usize { self.comm.n_threads }
    pub fn thread_id(&self) -> usize { self.thread_id }
    pub fn split(&self, n_way: usize) -> ThreadInfo<T> {
        let subcomm = self.comm.split(self.thread_id, n_way);
        ThreadInfo{ thread_id: self.thread_id % n_way, comm: subcomm }
    }
}

pub struct SpawnThreads<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> {
    child: S,
    n_threads: usize,
    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    SpawnThreads <T,At,Bt,Ct,S> {
    pub fn new(n_threads: usize, child: S) -> SpawnThreads<T, At, Bt, Ct, S>{
        SpawnThreads{ child: child, n_threads : n_threads,
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    pub fn set_n_threads(&mut self, n_threads: usize){ 
        self.n_threads = n_threads;
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for SpawnThreads<T, At, Bt, Ct, S> 
    where S: Send {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, _thr: &ThreadInfo<T>) -> () {
        //What we do here:
        //3. Spawn a bunch of threads. How many? 
        //  For now store these in cntl tree.
        //  This will be easy to change since this cntl tree will be the root (so the 'outside' has
        //  ownership.
        //4. Set cpu affinity for each thread.
        //1. Each thread gets an alias (shallow copy) of a, b, and c.
        //      Todo: make an unsafe function for matrices "get alias"
        //2. Each thread gets its own (deep copy) of the control tree.
        //      Todo: make an function for control tree "mirror"

        let global_comm : Arc<ThreadComm<T>> = Arc::new(ThreadComm::new(self.n_threads));
        
        crossbeam::scope(|scope| {
            for id in 0..self.n_threads {
                let mut my_a = a.make_alias();
                let mut my_b = b.make_alias();
                let mut my_c = c.make_alias();
                let mut my_tree = self.child.shadow();
                let my_comm  = global_comm.clone();
                
                scope.spawn(move || {
                    let thr = ThreadInfo{thread_id: id, comm: my_comm};
                    my_tree.run(&mut my_a, &mut my_b, &mut my_c, &thr);
                });
            }
        });
    }
    unsafe fn shadow(&self) -> Self where Self: Sized {
        SpawnThreads{ child: self.child.shadow(), 
                      n_threads : self.n_threads,
                      _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}

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
            work_id: info.thread_id / subcomm_n_threads }
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for ParallelM<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T>) -> () {
        //Split the thread communicator and create new thread infos
        let parallel_info = match self.par_inf {
            Some(ref x) => { x },
            None => { let blah = self.make_subinfo( thr );
                      self.par_inf = Option::Some( blah );
                      match self.par_inf {
                        Some(ref x) => { x },
                        _ => unreachable!(),
                      }
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
        let h_padding_save = a.get_logical_h_padding();
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
    unsafe fn shadow(&self) -> Self where Self: Sized {
        ParallelM{ n_threads: self.n_threads, child: self.child.shadow(), iota: self.iota,
            par_inf: Option::None,
            _t: PhantomData, _at: PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
}
