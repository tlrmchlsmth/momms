mod kernel_nm;
mod kernel_mn;
mod ukernel;
mod kernel_xsmm;

pub use self::kernel_nm::KernelNM;
pub use self::kernel_mn::KernelMN;
pub use self::ukernel::Ukernel;
pub use self::kernel_xsmm::{Xsmm,KernelXsmmA2};

//Private
mod ukernel_wrapper;
mod xsmm_wrapper;
