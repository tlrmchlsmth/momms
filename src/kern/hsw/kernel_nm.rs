use matrix::{Scalar,Mat,Hierarch,Matrix,RowPanelMatrix,ColumnPanelMatrix};
use core::marker::{PhantomData};
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{U1,U4,U6,U8,U12,Unsigned};

pub struct KernelNM<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
    _nrt: PhantomData<Nr>,
    _mrt: PhantomData<Mr>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>, Nr: Unsigned, Mr: Unsigned> 
    GemmNode<T, At, Bt, Ct> for 
    KernelNM<T, At, Bt, Ct, Nr, Mr> {
    #[inline(always)]
    default unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        panic!("Macrokernel general case not implemented!");
    }
    fn new( ) -> KernelNM<T, At, Bt, Ct, Nr, Mr> { 
        KernelNM{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData, _nrt: PhantomData, _mrt: PhantomData } 
    }
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        let mut desc = Vec::new();
        desc.push(AlgorithmStep::M{bsz: Mr::to_usize()});
        desc.push(AlgorithmStep::N{bsz: Nr::to_usize()});
        desc
    }  
}

type T = f64;
impl<K: Unsigned>
    GemmNode<T, Hierarch<T, U4, K,  U1, U4>,
                Hierarch<T, K,  U12, U12, U1>,
                Hierarch<T, U4, U12, U12, U1>> for
    KernelNM<T, Hierarch<T, U4, K,  U1, U4>,
                Hierarch<T, K,  U12, U12, U1>,
                Hierarch<T, U4, U12, U12, U1>, U12, U4> 
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Hierarch<T, U4, K, U1, U4>, b: &mut Hierarch<T, K, U12, U12, U1>, c: &mut Hierarch<T, U4, U12, U12, U1>, _thr: &ThreadInfo<T>){ 
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let c_nr_stride = c.block_stride_x(0) as isize;
        let mut c_jr = cp;

        let mut jr : isize = 0;
        while jr < n {
            let b_jr = bp.offset(jr * K::to_isize());
    
            let mut ir : isize = 0;
            while ir < m {
                let a_ir = ap.offset(ir * K::to_isize());
                let c_ir = c_jr.offset(ir * U12::to_isize());

                asm!("vzeroall");
                asm!("
                    prefetcht0 0*8($0)
                    prefetcht0 24*8($0)
                    "
                    : : "r"(c_ir));
                //ymm12: a
                //ymm13: b1
                //ymm14: b2
                //ymm15: b3
                let mut a_ind : isize = 0;
                let mut b_ind : isize = 0;
                for iter in 0..k{
                    asm!("
                        vmovapd ymm13, [$3 + 8*$1 + 0]
                        vmovapd ymm14, [$3 + 8*$1 + 32]
                        vmovapd ymm15, [$3 + 8*$1 + 64]

                        vmovapd ymm12, [$2 + 8*$0 + 0]
                        vfmadd231pd ymm0, ymm12, ymm13
                        vfmadd231pd ymm1, ymm12, ymm14
                        prefetcht0 [$2+$0+8*8]
                        vfmadd231pd ymm2, ymm12, ymm15
            
                        vpermpd ymm12, ymm12, 0xB1
                        vfmadd231pd ymm3, ymm12, ymm13
                        vfmadd231pd ymm4, ymm12, ymm14
                        vfmadd231pd ymm5, ymm12, ymm15
                        
                        vperm2f128 ymm12, ymm12, ymm12, 1
                        vfmadd231pd ymm6, ymm12, ymm13
                        vfmadd231pd ymm7, ymm12, ymm14
                        vfmadd231pd ymm8, ymm12, ymm15

                        vpermpd ymm12, ymm12, 0xB1
                        vfmadd231pd ymm9, ymm12, ymm13
                        vfmadd231pd ymm10, ymm12, ymm14
                        vfmadd231pd ymm11, ymm12, ymm15
                        " 
                        : : "r"(a_ind),"r"(b_ind),"r"(a_ir),"r"(b_jr) : "ymm12 ymm13 ymm14 ymm15" :"intel");
                        a_ind += 4;
                        b_ind += 12;

                }
                asm!("
                    vaddpd ymm0, ymm0, [$0+8*0]
                    vaddpd ymm1, ymm1, [$0+8*4]
                    vaddpd ymm2, ymm2, [$0+8*8]
                    vaddpd ymm3, ymm3, [$0+8*12]
                    vaddpd ymm4, ymm4, [$0+8*16]
                    vaddpd ymm5, ymm5, [$0+8*20]
                    vaddpd ymm6, ymm6, [$0+8*24]
                    vaddpd ymm7, ymm7, [$0+8*28]
                    vaddpd ymm8, ymm8, [$0+8*32]
                    vaddpd ymm9, ymm9, [$0+8*36]
                    vaddpd ymm10, ymm10, [$0+8*40]
                    vaddpd ymm11, ymm11, [$0+8*44]

                    vmovapd [$0+8*0], ymm0
                    vmovapd [$0+8*4], ymm1
                    vmovapd [$0+8*8], ymm2
                    vmovapd [$0+8*12], ymm3
                    vmovapd [$0+8*16], ymm4
                    vmovapd [$0+8*20], ymm5
                    vmovapd [$0+8*24], ymm6
                    vmovapd [$0+8*28], ymm7
                    vmovapd [$0+8*32], ymm8
                    vmovapd [$0+8*36], ymm9
                    vmovapd [$0+8*40], ymm10
                    vmovapd [$0+8*44], ymm11
                "
                : : "r"(c_ir) : "memory" :"intel");

                ir += U4::to_isize();
            }
            jr += U12::to_isize();
            c_jr = c_jr.offset(c_nr_stride);
        }
    }
}

impl<K: Unsigned>
    GemmNode<T, Hierarch<T, U12, K,  U1, U12>,
                Hierarch<T, K,  U4, U4, U1>,
                Hierarch<T, U12, U4, U4, U1>> for
    KernelNM<T, Hierarch<T, U12, K,  U1, U12>,
                Hierarch<T, K,  U4, U4, U1>,
                Hierarch<T, U12, U4, U4, U1>, U4, U12> 
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Hierarch<T, U12, K, U1, U12>, b: &mut Hierarch<T, K, U4, U4, U1>, c: &mut Hierarch<T, U12, U4, U4, U1>, _thr: &ThreadInfo<T>){ 
        panic!("asdf");
        let ap0 = a.get_mut_buffer();
        let bp0 = b.get_mut_buffer();
        let cp0 = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut jr : isize = 0;
        while jr < n {
            let bp = bp0.offset(jr * K::to_isize());
            let cpj = cp0.offset(c.block_stride_x(0) as isize * (jr/U4::to_isize()));
    
            let mut ir : isize = 0;
            while ir < m {
                let ap = ap0.offset(ir * K::to_isize());
                let cp = cpj.offset(ir * U4::to_isize());

                asm!("vzeroall");
                asm!("
                    prefetcht0 0*8($0)
                    prefetcht0 24*8($0)
                    "
                    : : "r"(cp));
                /*asm!("
                    prefetcht0 0*8($0)
                    prefetcht0 8*8($0)
                    prefetcht0 16*8($0)
                    prefetcht0 24*8($0)
                    prefetcht0 32*8($0)
                    prefetcht0 40*8($0)
                    "
                    : : "r"(cp));*/
                //ymm12: b
                //ymm13: a1
                //ymm14: a2
                //ymm15: a3
                let mut a_ind : isize = 0;
                let mut b_ind : isize = 0;
                for iter in 0..k{
                    asm!("
                        vmovapd ymm13, [$3 + 8*$1 + 0]
                        vmovapd ymm14, [$3 + 8*$1 + 32]
                        vmovapd ymm15, [$3 + 8*$1 + 64]

                        vmovapd ymm12, [$2 + 8*$0 + 0]
                        vfmadd231pd ymm0, ymm12, ymm13
                        vfmadd231pd ymm1, ymm12, ymm14
                        prefetcht0 [$2+$0+8*8]
                        vfmadd231pd ymm2, ymm12, ymm15
            
                        vpermpd ymm12, ymm12, 0xB1
                        vfmadd231pd ymm3, ymm12, ymm13
                        vfmadd231pd ymm4, ymm12, ymm14
                        vfmadd231pd ymm5, ymm12, ymm15
                        
                        vperm2f128 ymm12, ymm12, ymm12, 1
                        vfmadd231pd ymm6, ymm12, ymm13
                        vfmadd231pd ymm7, ymm12, ymm14
                        vfmadd231pd ymm8, ymm12, ymm15

                        vpermpd ymm12, ymm12, 0xB1
                        vfmadd231pd ymm9, ymm12, ymm13
                        vfmadd231pd ymm10, ymm12, ymm14
                        vfmadd231pd ymm11, ymm12, ymm15
                        " 
                        : : "r"(b_ind),"r"(a_ind),"r"(bp),"r"(ap) : "ymm12 ymm13 ymm14 ymm15" :"intel");
                        a_ind += 12;
                        b_ind += 4;

                }
                asm!("
                    vaddpd ymm0, ymm0, [$0+8*0]
                    vaddpd ymm3, ymm3, [$0+8*4]
                    vaddpd ymm6, ymm6, [$0+8*8]
                    vaddpd ymm9, ymm9, [$0+8*12]

                    vaddpd ymm1, ymm1, [$0+8*16]
                    vaddpd ymm4, ymm4, [$0+8*20]
                    vaddpd ymm7, ymm7, [$0+8*24]
                    vaddpd ymm10, ymm10, [$0+8*28]

                    vaddpd ymm2, ymm2, [$0+8*32]
                    vaddpd ymm5, ymm5, [$0+8*36]
                    vaddpd ymm8, ymm8, [$0+8*40]
                    vaddpd ymm11, ymm11, [$0+8*44]

                    vmovapd [$0+8*0], ymm0
                    vmovapd [$0+8*4], ymm3
                    vmovapd [$0+8*8], ymm6
                    vmovapd [$0+8*12], ymm9

                    vmovapd [$0+8*16], ymm1
                    vmovapd [$0+8*20], ymm4
                    vmovapd [$0+8*24], ymm7
                    vmovapd [$0+8*28], ymm10

                    vmovapd [$0+8*32], ymm2
                    vmovapd [$0+8*36], ymm5
                    vmovapd [$0+8*40], ymm8
                    vmovapd [$0+8*44], ymm11
                "
                : : "r"(cp) : "memory" :"intel");

                ir += U12::to_isize();
            }
            jr += U4::to_isize();
        }
    }
}

extern crate libc;
use self::libc::{ c_double, int64_t };

extern{
    fn bli_dgemm_asm_6x8 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}

impl<K: Unsigned>
    GemmNode<T, Hierarch<T, U6, K,  U1, U6>,
                Hierarch<T, K,  U8, U8, U1>,
                Hierarch<T, U6, U8, U8, U1>> for
    KernelNM<T, Hierarch<T, U6, K,  U1, U6>,
                Hierarch<T, K,  U8, U8, U1>,
                Hierarch<T, U6, U8, U8, U1>, U8, U6> 
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut Hierarch<T, U6, K, U1, U6>, b: &mut Hierarch<T, K, U8, U8, U1>, c: &mut Hierarch<T, U6, U8, U8, U1>, _thr: &ThreadInfo<T>){ 
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_nr_stride = c.block_stride_x(0) as isize;
        let mut c_jr = cp;

        let mut jr : isize = 0;
        while jr < n {
            let b_jr = bp.offset(jr * K::to_isize());
    
            let mut ir : isize = 0;
            while ir < m {
                let a_ir = ap.offset(ir * K::to_isize());
                let c_ir = c_jr.offset(ir * U8::to_isize());

                bli_dgemm_asm_6x8 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_ir as *mut c_double,
                    U8::to_isize() as int64_t, U1::to_isize() as int64_t );


                ir += U6::to_isize();
            }
            jr += U8::to_isize();
            c_jr = c_jr.offset(c_nr_stride);
        }
    }
}

impl
    GemmNode<T, RowPanelMatrix<T,U6>,
                ColumnPanelMatrix<T,U8>,
                Hierarch<T,U6,U8,U8,U1>> for 
    KernelNM<T, RowPanelMatrix<T,U6>,
                ColumnPanelMatrix<T,U8>,
                Hierarch<T,U6,U8,U8,U1>, U8, U6>
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut RowPanelMatrix<T,U6>, b: &mut ColumnPanelMatrix<T,U8>, c: &mut Hierarch<T,U6,U8,U8,U1>, _thr: &ThreadInfo<T>){ 
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_nr_stride = c.block_stride_x(0) as isize;

        let mut c_jr = cp;
        let mut b_jr = bp;
        let mut jr : isize = 0;
        while jr < n {
            
            let mut a_ir = ap;
            let mut ir : isize = 0;
            while ir < m {
                let c_ir = c_jr.offset(ir * U8::to_isize());

                bli_dgemm_asm_6x8 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_ir as *mut c_double,
                    U8::to_isize() as int64_t, U1::to_isize() as int64_t );


                ir += U6::to_isize();
                a_ir = a_ir.offset(a.get_panel_stride() as isize);
            }
            jr += U8::to_isize();
            c_jr = c_jr.offset(c_nr_stride);
            b_jr = b_jr.offset(b.get_panel_stride() as isize);
        }
    }
}

impl GemmNode<T, RowPanelMatrix<T,U6>,
                 ColumnPanelMatrix<T,U8>,
                 Matrix<T>> for 
    KernelNM<T, RowPanelMatrix<T,U6>,
                ColumnPanelMatrix<T,U8>,
                Matrix<T>, U8, U6>
{
    #[inline(always)]
    unsafe fn run(&mut self, a: &mut RowPanelMatrix<T,U6>, b: &mut ColumnPanelMatrix<T,U8>, c: &mut Matrix<T>, _thr: &ThreadInfo<T>){ 
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let m = c.height() as isize;
        let n = c.width() as isize;
        let k = a.width() as isize;

        let mut alpha : f64 = 1.0;
        let mut beta : f64 = 1.0;

        let c_rs = c.get_row_stride() as i64;
        let c_cs = c.get_column_stride() as i64;
        let c_nr_stride = U8::to_i64() * c_cs;
        let c_mr_stride = U6::to_i64() * c_rs;

        let mut c_jr = cp;
        let mut b_jr = bp;
        let mut jr : isize = 0;
        while jr < n {
            
            let mut a_ir = ap;
            let mut c_ir = c_jr;
            let mut ir : isize = 0;
            while ir < m {

                bli_dgemm_asm_6x8 (
                    k as int64_t,
                    &mut alpha as *mut c_double,
                    a_ir as *mut c_double,
                    b_jr as *mut c_double,
                    &mut beta as *mut c_double,
                    c_ir as *mut c_double,
                    c_rs, c_cs );


                ir += U6::to_isize();
                a_ir = a_ir.offset(a.get_panel_stride() as isize);
                c_ir = c_ir.offset(c_mr_stride as isize);
            }
            jr += U8::to_isize();
            c_jr = c_jr.offset(c_nr_stride as isize);
            b_jr = b_jr.offset(b.get_panel_stride() as isize);
        }
    }
}
