/*use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;*/

fn main() -> () {

    if cfg!(feature="blis") {
        //Link with BLIS assuming it is installed to the default location:
        println!("cargo:rustc-link-search=native={}/blis/lib", std::env::home_dir().unwrap().to_str().unwrap() );
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-lib=static=blis");
        
        //Needed when BLIS is compiled with GCC and OpenMP:
        println!("cargo:rustc-link-search=native=/usr/lib/gcc/x86_64-linux-gnu/5");
        println!("cargo:rustc-link-lib=dylib=gomp");
        

        //Needed for linking with BLIS when it was compiled with icc
        /*
        println!("cargo:rustc-link-lib=static=irc");
        */
    }

    if cfg!(feature="libxsmm"){
        println!("cargo:rustc-link-search=native={}/libxsmm/lib", std::env::home_dir().unwrap().to_str().unwrap() );
        println!("cargo:rustc-link-lib=static=xsmm");
    }
}
