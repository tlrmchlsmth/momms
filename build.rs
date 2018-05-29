extern crate bindgen;

use std::{sync::{Arc, RwLock},collections::HashSet};
use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};


 #[derive(Debug)]
 struct MacroCallback {
//     macros: Arc<RwLock<HashSet<String>>>,
 }
 impl ParseCallbacks for MacroCallback {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
 //       self.macros.write().unwrap().insert(name.into());
        if ["FP_SUBNORMAL", "FP_NORMAL", "FP_ZERO", "FP_INFINITE", "FP_NAN"].contains(&name) {
            return MacroParsingBehavior::Ignore
        }
        MacroParsingBehavior::Default
     }
 }

fn main() -> () {

    if cfg!(feature="blis") {
        //Link with BLIS assuming it is installed to the default location:
        println!("cargo:rustc-link-search=native={}/blis/lib", std::env::home_dir().unwrap().to_str().unwrap() );
        println!("cargo:rustc-link-search=native=/usr/local/lib");
        println!("cargo:rustc-link-lib=static=blis");
        
        //Needed when BLIS is compiled with GCC and OpenMP:
//        println!("cargo:rustc-link-search=native=/usr/lib/gcc/x86_64-linux-gnu/5");
//        println!("cargo:rustc-link-lib=dylib=gomp");

        //Use bindgen to create bindings to BLIS's typedefs
        //This allows us to interface with the BLIS micro-kernel
		let bindings = bindgen::Builder::default()
			.header("blis_types_wrapper.h")
            .clang_arg("-include")
            .clang_arg("stddef.h")
            .clang_arg(format!("-I/{}/blis/include/blis", std::env::home_dir().unwrap().to_str().unwrap()))
            .parse_callbacks(Box::new(MacroCallback{ }))
            .whitelist_type("pack_t*")
            .whitelist_type("auxinfo_t")
			.generate()
			.expect("Unable to generate bindings");

        // Write the bindings to the $OUT_DIR/bindings.rs file.
    	let out_path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    	bindings
			.write_to_file(out_path.join("bindings.rs"))
			.expect("Couldn't write bindings!");

        //Needed for linking with BLIS when it was compiled with icc
//        println!("cargo:rustc-link-lib=static=irc");
        
    }

    if cfg!(feature="libxsmm"){
        println!("cargo:rustc-link-search=native={}/libxsmm/lib", std::env::home_dir().unwrap().to_str().unwrap() );
        println!("cargo:rustc-link-lib=static=xsmm");
    }
}
