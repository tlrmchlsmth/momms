#![feature(specialization)]
#![feature(alloc, allocator_api)]
#![feature(asm)]

#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]
#![allow(unknown_lints)]
#![allow(inline_always)] 
#![allow(too_many_arguments)]
#![allow(many_single_char_names)]

extern crate core;
extern crate typenum;
extern crate libc;

pub mod matrix;
pub mod composables;
pub mod thread_comm;
pub mod kern;
pub mod util;
