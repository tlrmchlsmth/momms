/*use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;*/

fn main() -> () {
    println!("cargo:rustc-link-search=native={}/blis/lib", std::env::home_dir().unwrap().to_str().unwrap() );
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=blis");

//  The following is needed for compiling with mkl
//    println!("cargo:rustc-link-lib=static=irc");
}
