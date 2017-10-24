# momms

Introduction
------------

MOMMS is a sandbox for implementing algorithms for matrix-matrix multiplicaiton (MMM) written in Rust.
Algorithms are instantiated via type composition.

How-to
------
Get Rust Nightly:
https://www.rustup.rs/

Install BLIS to your home directory:
https://github.com/flame/blis

Get and install hwloc:
https://www.open-mpi.org/projects/hwloc/

To build MOMMS binaries in release using the blis kernel:
cargo build --release --features "blis hsw"


Funding
-------
This project and the research associated with it was sponsored in part by an Intel Parallel Computing Center grant 
as well as a grant from the National Science Foundation (Award ACI-1550493).

_Any opinions, findings and conclusions or recommendations expressed in this
material are those of the author(s) and do not necessarily reflect the views of
the National Science Foundation (NSF)._

