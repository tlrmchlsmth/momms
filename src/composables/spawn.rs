//extern crate scoped_threadpool;
extern crate threadpool;
extern crate thread_local;
extern crate hwloc;

use matrix::{Scalar,Mat};
use core::marker::{PhantomData};
use thread_comm::{ThreadComm,ThreadInfo};
use composables::{GemmNode,AlgorithmStep};

use std::sync::{Arc,Mutex};
use std::cell::{RefCell};
use self::threadpool::ThreadPool;
use self::thread_local::ThreadLocal;
use self::hwloc::{Topology, ObjectType, CPUBIND_THREAD, CpuSet};
use libc;

fn cpuset_for_core(topology: &Topology, idx: usize) -> CpuSet {
    let cores = (*topology).objects_with_type(&ObjectType::Core).unwrap();
    match cores.get(idx) {
        Some(val) => val.cpuset().unwrap(),
        None => panic!("No Core found with id {}", idx)
    }
}

pub struct SpawnThreads<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    where S: Send, T: 'static, S: 'static, At: 'static, Bt: 'static, Ct: 'static {
    n_threads: usize,
    pool: ThreadPool,

    cntl_cache: Arc<ThreadLocal<RefCell<S>>>,

    _t: PhantomData<T>,
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
}
impl<T: Scalar,At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>> 
    SpawnThreads <T,At,Bt,Ct,S> 
    where S: Send {
    pub fn set_n_threads(&mut self, n_threads: usize){ 
        //Create new thread pool
        self.n_threads = n_threads;
        if n_threads > 1 {
            self.pool = ThreadPool::new(n_threads-1);
        } else {
            self.pool = ThreadPool::new(1);
        }

        //Clear the control tree cache
        Arc::get_mut(&mut self.cntl_cache).expect("").clear();
        
        //Bind threads to cores
        self.bind_threads();
    }
    fn bind_threads(&mut self) {
        //Get topology
        let topo = Arc::new(Mutex::new(Topology::new()));
        let comm : Arc<ThreadComm<T>> = Arc::new(ThreadComm::new(self.n_threads));

        //Bind workers to cores.
        for id in 1..self.n_threads {
            let my_topo = topo.clone();
            let my_comm  = comm.clone();
            self.pool.execute(move || {
                let tid = unsafe { libc::pthread_self() };
                {
                    let mut locked_topo = my_topo.lock().unwrap();
                    //let before = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
                    let bind_to = cpuset_for_core(&*locked_topo, id);
                    let _ = locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD);
                    //let after = locked_topo.get_cpubind_for_thread(tid, CPUBIND_THREAD);
                }
                //Barrier to make sure therad binding is done.
                let thr = ThreadInfo::new(id, my_comm);
                thr.barrier()
            });
        }
        
        //Bind parent to a core.
        let tid = unsafe { libc::pthread_self() };
        {
            let mut locked_topo = topo.lock().unwrap();
            let bind_to = cpuset_for_core(&*locked_topo, 0);
            let _ = locked_topo.set_cpubind_for_thread(tid, bind_to, CPUBIND_THREAD);
        }
        let thr = ThreadInfo::new(0, comm);
        thr.barrier();
    }
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, S: GemmNode<T, At, Bt, Ct>>
    GemmNode<T, At, Bt, Ct> for SpawnThreads<T, At, Bt, Ct, S> 
    where S: Send {
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut At, b: &mut Bt, c:&mut Ct, _thr: &ThreadInfo<T>) -> () {
        //Create global thread communicator
        let comm : Arc<ThreadComm<T>> = Arc::new(ThreadComm::new(self.n_threads));

        //Make some shallow copies here to pass into the scoped,
        //because self.pool borrows self as mutable
        //let cache = self.cntl_cache.clone();
    
        //Spawn n-1 workers since head thread will do work too.
        for id in 1..self.n_threads {
            //Make some shallow copies because of borrow rules
            let mut my_a = a.make_alias();
            let mut my_b = b.make_alias();
            let mut my_c = c.make_alias();
            let my_comm  = comm.clone();
            let my_cache = self.cntl_cache.clone();

            self.pool.execute(move || {
                //Make this thread's communicator holder
                let thr = ThreadInfo::new(id, my_comm);

                //Read this thread's cached control tree
                let cntl_tree_cell = my_cache.get_or(|| Box::new(RefCell::new(S::new())));

                //Run subproblem
                cntl_tree_cell.borrow_mut().run(&mut my_a, &mut my_b, &mut my_c, &thr);
                thr.barrier();
            });
        }

        //Do parent thread's work
        let thr = ThreadInfo::new(0, comm);
        let cntl_tree_cell = self.cntl_cache.get_or(|| Box::new(RefCell::new(S::new())));
        cntl_tree_cell.borrow_mut().run(a, b, c, &thr);
        thr.barrier();
    }
    fn new() -> Self {
        SpawnThreads{ n_threads : 1, pool: ThreadPool::new(1),
                 cntl_cache: Arc::new(ThreadLocal::new()),
                 _t: PhantomData, _at:PhantomData, _bt: PhantomData, _ct: PhantomData }
    }
    fn hierarchy_description() -> Vec<AlgorithmStep> {
        S::hierarchy_description()
    }
}

