extern crate alloc;
use std::ptr::{self};
use std::mem;
use self::alloc::heap;
use std::sync::{Arc,Barrier};
use std::sync::atomic::{AtomicPtr,AtomicUsize,Ordering};

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

    barrier: Barrier,

    //I guess subcomms needs to have interor mutability?
    sub_comms: Vec<AtomicPtr<Arc<ThreadComm<T>>>>,
}
impl<T> ThreadComm<T> {
    fn new( n_threads: usize ) -> ThreadComm<T> { 
        let init_ptr: *const T = ptr::null();
        let mut sub_comms = Vec::with_capacity(n_threads);
        for i in 0..n_threads {
            let ptr: *const Arc<ThreadComm<T>> = ptr::null();
            sub_comms.push( AtomicPtr::new(ptr as *mut Arc<ThreadComm<T>>) );
        }
        ThreadComm{ n_threads: n_threads,
            slot: AtomicPtr::new( init_ptr as *mut T),
            slot_reads: AtomicUsize::new(n_threads),
            barrier: Barrier::new(n_threads),
            sub_comms: sub_comms,
        }
    }

    fn barrier( &self, _info: &ThreadInfo<T> ) {
        self.barrier.wait();
    }

    fn broadcast( &self, info: &ThreadInfo<T>, to_send: *mut T ) -> *mut T {
        if info.thread_id == 0 {
            //Spin while waiting for the thread communicator to be ready to broadcast
            while self.slot_reads.load( Ordering::Relaxed ) != self.n_threads {}
            self.slot.store( to_send, Ordering::Relaxed );
            self.slot_reads.store( 0, Ordering::Relaxed ); 
        }
        //Spin while waiting for the thread communicator chief to broadcast
        while self.slot_reads.load( Ordering::Relaxed ) == self.n_threads {}
        self.slot_reads.fetch_add( 1, Ordering::Relaxed );
        self.slot.load( Ordering::Relaxed )
    }
    //Pretty sure with this implementation, split can only be called one time.
    fn split( &self, info: &ThreadInfo<T>, n_way: usize ) -> Arc<ThreadComm<T>> {
        assert!( self.n_threads % n_way == 0 );
        let sub_comm_number = info.thread_id / n_way; // Which subcomm are we going to use?
        let sub_comm_id = info.thread_id % n_way; // What is our id within the subcomm?

        if sub_comm_id == 0 {
            if !self.sub_comms[ sub_comm_number ].load( Ordering::Relaxed ).is_null() {
                self.sub_comms[ sub_comm_number ].store( 
                    &mut Arc::new(ThreadComm::new( self.n_threads )) as *mut Arc<ThreadComm<T>>,
                    Ordering::Relaxed );
            }
        }
        while self.sub_comms[ sub_comm_number ].load( Ordering::Relaxed ).is_null() {}
        unsafe{
            let blah = self.sub_comms[sub_comm_number].load( Ordering::Relaxed );
            (*blah).clone()
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
    pub fn single_thread( ) -> ThreadInfo<T>{
        ThreadInfo{ thread_id : 0, comm : Arc::new(ThreadComm::new(1)) }
    }
    pub fn new( thread_id: usize, comm: Arc<ThreadComm<T>> ) -> ThreadInfo<T> {
        ThreadInfo{ thread_id : thread_id, comm : comm.clone() }
    }
    pub fn barrier( &self ) {
        self.comm.barrier(&self);
    }
    pub fn broadcast( &self, to_send: *mut T ) -> *mut T {
        self.comm.broadcast(&self, to_send)
    }
}

struct MatrixBuffer {
    buf: *mut f64,
    len: usize,
}

impl MatrixBuffer {
    fn new( len: usize ) -> MatrixBuffer {
        unsafe {
            let buf = heap::allocate( len * mem::size_of::<f64>(), 4096 ) as *mut f64;
            MatrixBuffer{ buf: buf, len: len }
        }
    }
    fn set( &mut self, id: usize, val: f64  ) {
        unsafe{
            ptr::write( self.buf.offset(id as isize), val ); 
        }
    }
    fn get( &self, id: usize ) -> f64 {
        unsafe{
            ptr::read( self.buf.offset(id as isize) )
        }
    }
    fn from( &mut self, other: MatrixBuffer ) {
        self.buf = other.buf;
        self.len = other.len;
    }
    fn get_alias( &self ) -> MatrixBuffer {
        MatrixBuffer{ buf: self.buf, len: self.len }
    }
}
unsafe impl Sync for MatrixBuffer {}
unsafe impl Send for MatrixBuffer {}

pub fn blah() {
    let mut mat = MatrixBuffer::new(2);
    mat.set(0, 0.0);
    mat.set(1, 0.0);

    let global_comm = Arc::new(ThreadComm::new( 2 ));

    crossbeam::scope(|scope| {
        for id in 0..2 {
            let mut my_alias = mat.get_alias();
            let mut my_comm  = global_comm.clone();
            scope.spawn(move || {
                //let mut my_alias = ref mat.get_alias();
                let info = ThreadInfo{thread_id: id, comm: my_comm};

                let mat_inside = MatrixBuffer::new(2);

                println!("tid {} broadcasting!", info.thread_id);
                let mut curr_matrix = MatrixBuffer::new(0);
                let ptr = info.broadcast( mat_inside.buf );
                curr_matrix.buf = ptr;
                curr_matrix.set(info.thread_id, 4.0 );

                println!("tid {} barriering!", info.thread_id);
                info.barrier();
                println!("tid {} setting!", info.thread_id );
                if info.thread_id == 0 {
                    print!("{} ", curr_matrix.get( 0 ) );
                    print!("{} ", curr_matrix.get( 1 ) );
                }

                my_alias.set( info.thread_id, 5.0 );
                println!("tid {} done!", info.thread_id );
            });
        }
    });
        
    for id in 0..2 {
        print!("{} ", mat.get( id ) );
    }
    println!("");
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
    pub fn new( child: S, n_threads: usize ) -> SpawnThreads<T, At, Bt, Ct, S>{
        SpawnThreads{ child: child, n_threads : n_threads,
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    pub fn set_n_threads( &mut self, n_threads: usize ){ 
        self.n_threads = n_threads;
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for SpawnThreads<T, At, Bt, Ct, S> {
    #[inline(always)]
    unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c:&mut Ct, thr: &ThreadInfo<T> ) -> () {
        //What we do here:
        //3. Spawn a bunch of threads. How many? 
        //  For now store these in cntl tree.
        //  This will be easy to change since this cntl tree will be the root (so the 'outside' has
        //  easy ownership.
        //4. Set cpu affinity for each thread.
        //1. Each thread gets an alias (shallow copy) of a, b, and c.
        //      Todo: make an unsafe function for matrices "get alias"
        //2. Each thread gets its own (deep copy) of the control tree.
        //      Todo: make an function for control tree "mirror"
/*
        self.a_pack.resize( a.height(), a.width() );
        self.packer.pack( a, &mut self.a_pack );
        self.child.run(a, b, c);

    crossbeam::scope(|scope| {
        for id in 0..2 {
            let mut my_alias = mat.get_alias();
            let mut my_comm  = global_comm.clone();
            scope.spawn(move || {
                //let mut my_alias = ref mat.get_alias();
                let info = ThreadInfo{thread_id: id, comm: my_comm};

                let mat_inside = MatrixBuffer::new(2);

                println!("tid {} broadcasting!", info.thread_id);
                let mut curr_matrix = MatrixBuffer::new(0);
                let ptr = info.broadcast( mat_inside.buf );
                curr_matrix.buf = ptr;
                curr_matrix.set(info.thread_id, 4.0 );

                println!("tid {} barriering!", info.thread_id);
                info.barrier();
                println!("tid {} setting!", info.thread_id );
                if info.thread_id == 0 {
                    print!("{} ", curr_matrix.get( 0 ) );
                    print!("{} ", curr_matrix.get( 1 ) );
                }

                my_alias.set( info.thread_id, 5.0 );
                println!("tid {} done!", info.thread_id );
            });
        }
    });*/
    }
}
