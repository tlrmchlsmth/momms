extern crate alloc;
use std::ptr::{self};
use std::sync::{Arc,RwLock};
//use std::sync::{Barrier};
use std::sync::atomic::{AtomicPtr,AtomicUsize,AtomicBool,Ordering};

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
    pub fn new(n_threads: usize) -> ThreadComm<T> { 
        let init_ptr: *const T = ptr::null();

        let mut sub_comms = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
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
            while self.barrier_sense.load(Ordering::Relaxed) == my_sense { }
        } 

        //self.barrier.wait();
    }

    fn broadcast(&self, info: &ThreadInfo<T>, to_send: *mut T) -> *mut T {
        
        if info.thread_id == 0 {
            //Spin while waiting for the thread communicator to be ready to broadcast
            while self.slot_reads.load(Ordering::Relaxed) != self.n_threads {
            }
            self.slot.store(to_send, Ordering::Relaxed);
            self.slot_reads.store(0, Ordering::Relaxed); 
        }
        //Spin while waiting for the thread communicator chief to broadcast
        while self.slot_reads.load(Ordering::Relaxed) == self.n_threads {
        }
        self.slot_reads.fetch_add(1, Ordering::Relaxed);
        self.slot.load(Ordering::Relaxed)
        /*
        self.barrier(info.thread_id);
        if info.thread_id == 0 { 
            self.slot.store(to_send, Ordering::Relaxed);
        }
        self.barrier(info.thread_id);
        self.slot.load(Ordering::Relaxed)*/
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
        comm.unwrap()
    }
}
//unsafe impl Sync for ThreadComm {}
//unsafe impl Send for ThreadComm {}

pub struct ThreadInfo<T> {
    thread_id: usize,
    comm: Arc<ThreadComm<T>>,
}
impl<T> ThreadInfo<T> {
    pub fn new( id: usize, comm: Arc<ThreadComm<T>> ) -> ThreadInfo<T> {
        ThreadInfo{ thread_id: id, comm: comm }
    }
    pub fn single_thread() -> ThreadInfo<T>{
        ThreadInfo{ thread_id : 0, comm : Arc::new(ThreadComm::new(1)) }
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
        let subcomm_id = self.thread_id % (self.comm.n_threads / n_way);
        ThreadInfo{ thread_id: subcomm_id, comm: subcomm }
    }
}
