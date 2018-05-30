extern crate bindgen;

use std::{env, process::Command, path::Path};
use bindgen::callbacks::{MacroParsingBehavior, ParseCallbacks};


 #[derive(Debug)]
 struct MacroCallback { }
 impl ParseCallbacks for MacroCallback {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        if ["FP_SUBNORMAL", "FP_NORMAL", "FP_ZERO", "FP_INFINITE", "FP_NAN"].contains(&name) {
            return MacroParsingBehavior::Ignore
        }
        MacroParsingBehavior::Default
     }
 }

fn main() -> () {
    let out_dir = env::var("OUT_DIR").unwrap();

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
    	let out_path = std::path::PathBuf::from(&out_dir);
    	bindings
			.write_to_file(out_path.join("bindings.rs"))
			.expect("Couldn't write bindings!");

        //
        // Compile knm "micro-kernel" standalone
        // via: icc -I${HOME}/blis/include/blis -march=knm -O3 -std=c11 -c sgemm_knm_int_24x16.c -o sgemm_knm_int_24x16.o
        //
        if cfg!(feature="knm"){
            Command::new("icc").arg(&format!("-I{}/blis/include/blis", std::env::home_dir().unwrap().to_str().unwrap()))
                               .args(&["-march=knm", "-O3", "-std=c11", "-fPIC", "-c", "sgemm_knm_int_24x16.c", "-o"])
                               .arg(&format!("{}/sgemm_knm_int_24x16.o", &out_dir))
                               .status().unwrap();

            Command::new("icc").arg(&format!("-I{}/blis/include/blis", std::env::home_dir().unwrap().to_str().unwrap()))
                               .args(&["-march=knm", "-O3", "-std=c11", "-fPIC", "-c", "sgemm_knm_asm_24x16.c", "-o"])
                               .arg(&format!("{}/sgemm_knm_asm_24x16.o", &out_dir))
                               .status().unwrap();

            Command::new("ar").args(&["crus", "libknmkernel.a", "sgemm_knm_int_24x16.o", "sgemm_knm_asm_24x16.o"])
                              .current_dir(&Path::new(&out_dir))
                              .status().unwrap();
            println!("cargo:rustc-link-search=native={}", &out_dir);
            println!("cargo:rustc-link-lib=static=knmkernel");
//            
        }

        //Needed for linking with BLIS when it was compiled with icc
//        println!("cargo:rustc-link-lib=static=irc");
        
    }

    if cfg!(feature="libxsmm"){
        println!("cargo:rustc-link-search=native={}/libxsmm/lib", std::env::home_dir().unwrap().to_str().unwrap() );
        println!("cargo:rustc-link-lib=static=xsmm");
    }
}
