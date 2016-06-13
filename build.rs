use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() -> () {
    println!("cargo:rustc-link-search=native=/Users/tyler/blis/lib");
    println!("cargo:rustc-link-lib=static=blis");
}
