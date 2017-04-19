#![feature(specialization)]
#![feature(alloc, heap_api)]
#![feature(conservative_impl_trait)]
#![feature(cfg_target_feature)]
#![feature(asm)]

#![allow(dead_code)]

extern crate core;
extern crate typenum;
extern crate hwloc;
extern crate libc;

pub mod matrix;
pub mod composables;
pub mod thread_comm;
pub mod kern;
pub mod util;
