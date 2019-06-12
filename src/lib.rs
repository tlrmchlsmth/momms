#![feature(specialization)]
#![feature(const_generics)]
#![feature(allocator_api)]
#![feature(asm)]

extern crate core;
extern crate libc;

pub mod matrix;
pub mod composables;
pub mod thread_comm;
pub mod kern;
pub mod util;
