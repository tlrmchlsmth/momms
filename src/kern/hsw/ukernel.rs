use matrix::{Scalar,Mat,ColumnPanelMatrix,RowPanelMatrix,Matrix,Hierarch};
use core::marker::{PhantomData};
use core::ptr;
use composables::{GemmNode,AlgorithmStep};
use thread_comm::{ThreadInfo};
use typenum::{Unsigned,U1,U4,U6,U8,U12};
use std::fmt;
use std::ops::{Add,AddAssign,Mul};

extern crate libc;
use self::libc::{ c_double, int64_t };
extern{
    fn bli_dgemm_asm_6x8 ( k: int64_t,
        alpha: *mut c_double, a: *mut c_double, b: *mut c_double, beta: *mut c_double, 
        c: *mut c_double, rs_c: int64_t, cs_c: int64_t ) -> ();
}

pub struct Ukernel<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>>{
    _at: PhantomData<At>,
    _bt: PhantomData<Bt>,
    _ct: PhantomData<Ct>,
    _t: PhantomData<T>,
}
impl<T: Scalar, At: Mat<T>, Bt: Mat<T>, Ct: Mat<T>> 
    GemmNode<T, At, Bt, Ct> for Ukernel<T, At, Bt, Ct> {
    #[inline(always)]
    default unsafe fn run( &mut self, a: &mut At, b: &mut Bt, c: &mut Ct, _thr: &ThreadInfo<T> ) -> () {
        for z in 0..a.width() {
            for y in 0..c.height() {
                for x in 0..c.width() {
                    let t = a.get(y,z) * b.get(z,x) + c.get(y,x);
                    c.set( y, x, t );
                }   
            }   
        }   
    }   
    fn new( ) -> Ukernel<T, At, Bt, Ct> { 
        Ukernel{ _at: PhantomData, _bt: PhantomData, _ct: PhantomData, _t: PhantomData } 
    }   
    fn hierarchy_description( ) -> Vec<AlgorithmStep> {
        Vec::new()
    }   
}

impl GemmNode<f64, RowPanelMatrix<f64, U6>, ColumnPanelMatrix<f64, U8>, Matrix<f64>> 
    for Ukernel<f64, RowPanelMatrix<f64, U6>, ColumnPanelMatrix<f64, U8>, Matrix<f64>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut RowPanelMatrix<f64, U6>, 
                   b: &mut ColumnPanelMatrix<f64, U8>, 
                   c: &mut Matrix<f64>, 
                   _thr: &ThreadInfo<f64> ) -> () {
        debug_assert!(c.height() <= U6::to_usize());
        debug_assert!(c.width() <= U8::to_usize());


        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();
        let rs_c = c.get_row_stride();
        let cs_c = c.get_column_stride();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        if c.height() == U6::to_usize() && c.width() == U8::to_usize() {
            //bli_dgemm_asm_8x4 ( 
            bli_dgemm_asm_6x8 (
                a.width() as int64_t,
                &mut alpha as *mut c_double,
                ap as *mut c_double,
                bp as *mut c_double,
                &mut beta as *mut c_double,
                cp as *mut c_double,
                rs_c as int64_t, cs_c as int64_t );
        }
        else {
            //TODO: cache c_tmp somewhere!
            let mut t : Matrix<f64> = Matrix::new( U6::to_usize(), U8::to_usize() );
            let tp = t.get_mut_buffer();
            let rs_t = t.get_row_stride();
            let cs_t = t.get_column_stride();
            beta = 0.0;

            bli_dgemm_asm_6x8 (
                a.width() as int64_t,
                &mut alpha as *mut c_double,
                ap as *mut c_double,
                bp as *mut c_double,
                &mut beta as *mut c_double,
                tp as *mut c_double,
                rs_t as int64_t, cs_t as int64_t );
    

            t.push_y_view(c.height());
            t.push_x_view(c.width());
            c.axpby_small( 1.0, &t, 1.0 );
        }
    }
}

impl<K: Unsigned>
    GemmNode<f64, Hierarch<f64, U6, K, U1, U6>,
                  Hierarch<f64, K, U8, U8, U1>,
                  Hierarch<f64, U6, U8, U8, U1>> for
    Ukernel<f64, Hierarch<f64, U6, K, U1, U6>,
                 Hierarch<f64, K, U8, U8, U1>,
                 Hierarch<f64, U6, U8, U8, U1>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut Hierarch<f64, U6, K, U1, U6>,
                   b: &mut Hierarch<f64, K, U8, U8, U1>,
                   c: &mut Hierarch<f64, U6, U8, U8, U1>,
                   _thr: &ThreadInfo<f64> ) -> () {
//        assert!(c.height() == self.mr);
//        assert!(c.width() == self.nr);


        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();
//        let rs_c = c.get_row_stride();
//        let cs_c = c.get_column_stride();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        //bli_dgemm_asm_8x4 ( 
        bli_dgemm_asm_6x8 (
            a.width() as int64_t,
            &mut alpha as *mut c_double,
            ap as *mut c_double,
            bp as *mut c_double,
            &mut beta as *mut c_double,
            cp as *mut c_double,
            8 as int64_t, 1 as int64_t );
    }
}

impl GemmNode<f64, RowPanelMatrix<f64, U6>, 
                   ColumnPanelMatrix<f64, U8>, 
                   Hierarch<f64, U6, U8, U8, U1>> for
    Ukernel<f64, RowPanelMatrix<f64, U6>, 
                 ColumnPanelMatrix<f64, U8>, 
                 Hierarch<f64, U6, U8, U8, U1>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut RowPanelMatrix<f64, U6>, 
                   b: &mut ColumnPanelMatrix<f64, U8>, 
                   c: &mut Hierarch<f64, U6, U8, U8, U1>,
                   _thr: &ThreadInfo<f64> ) -> () {
//        assert!(c.height() == self.mr);
//        assert!(c.width() == self.nr);
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();
//        let rs_c = c.get_row_stride();
//        let cs_c = c.get_column_stride();
        let mut alpha: f64 = 1.0;
        let mut beta: f64 = 1.0;

        //bli_dgemm_asm_8x4 ( 
        bli_dgemm_asm_6x8 (
            a.width() as int64_t,
            &mut alpha as *mut c_double,
            ap as *mut c_double,
            bp as *mut c_double,
            &mut beta as *mut c_double,
            cp as *mut c_double,
            8 as int64_t, 1 as int64_t );
    }
}
/*extern crate simd;
#[cfg(target_feature = "avx")]
use self::simd::x86::avx::f64x4;*/
/*
extern "platform-intrinsic" {
    //fn x86_mm256_permutevar_pd(u: f64x4, v: i64x4) -> f64x4;
//    fn simd_shuffle4<T: Simd, U: Simd<Elem = T::Elem>>(x: T, y: T, idx: [u32; 4]) -> U;
    fn simd_shuffle4<T, U>(x: T, y: T, idx: [u32; 4]) -> U;
}
extern {
    #[link_name = "llvm.x86.avx.vperm2f128.pd.256"]
    fn x86_mm256_perm2f128_pd(u: f64x4, v: f64x4, imm: i8) -> f64x4;
//    #[link_name = "llvm.x86.avx.permute_pd.256"]
//    fn x86_mm256_permute_pd(u: f64x4, imm: i8) -> f64x4;
//    #[link_name = "llvm.x86.avx.builtin.shufflevector.256"]
//    fn x86_mm256_permute_pd(u: f64x4, i0: u8, i1: u8, i2: u8, i3: u8) -> f64x4;
    #[link_name = "llvm.x86.avx.blend.pd.256"]
    fn x86_mm256_blend_pd(u: f64x4, v: f64x4, imm: i8) -> f64x4;
//    #[link_name = "llvm.x86.avx.load.pd.256"]
//    fn x86_mm256_load_pd(u: *const f64x4) -> f64x4;
//    #[link_name = "llvm.x86.avx.store.pd.256"]
//    fn x86_mm256_store_pd(u: *mut f64x4, v: f64x4);
    #[link_name = "llvm.x86.fma.vfmadd.pd.256"]
    fn x86_mm256_fmadd_pd(a: f64x4, b: f64x4, c: f64x4) -> f64x4;
}

#[repr(simd)]
#[derive(Copy,Clone)]
struct f64x4(f64, f64, f64, f64);
impl f64x4 {
    #[inline(always)]
    fn zero() -> f64x4 {
        f64x4(0.0, 0.0, 0.0, 0.0)
    }
    #[inline(always)]
    fn fma(a: f64x4, b: f64x4, c: &mut f64x4) {
        unsafe {
            let tmp = x86_mm256_fmadd_pd(a, b, *c);
            *c = tmp;
        }
    }
    #[inline(always)]
    fn swiz_inner(u: f64x4) -> f64x4 {
        unsafe {
            //swap u ``within lanes'' so that:
            //u.0 := u.1
            //u.1 := u.0
            //u.2 := u.3
            //u.3 := u.2
            simd_shuffle4(u, u, [1, 0, 3, 2])
        }
    }
    #[inline(always)]
    fn swiz_outer(u: f64x4) -> f64x4 {
        unsafe {
            //swap u ``within lanes'' so that:
            //u.0 := u.2
            //u.1 := u.3
            //u.2 := u.0
            //u.3 := u.1
            x86_mm256_perm2f128_pd(u, u, 0b0000_0011_i8)
            //simd_shuffle4(u, u, [2,3,0,1])
        }
    }
    #[inline(always)]
    fn blend_outer(u: f64x4, v: f64x4) -> (f64x4, f64x4) {
        unsafe {
            (simd_shuffle4(u, v, [0, 1, 6, 7]),
            simd_shuffle4(u, v, [4, 5, 2, 3]))
        }
    }
    #[inline(always)]
    fn blend_inner(u: f64x4, v: f64x4) -> (f64x4, f64x4) {
        unsafe {
            (simd_shuffle4(u, v, [0, 5, 2, 7]),
            simd_shuffle4(u, v, [4, 1, 6, 3]))
        }
    }
}
impl Mul for f64x4{
    type Output = f64x4;
    fn mul(self, rhs: f64x4) -> f64x4 {
        f64x4(self.0 * rhs.0, self.1 * rhs.1, self.2 * rhs.2, self.3 * rhs.3)
    }
}
impl Add for f64x4 {
    type Output = f64x4;
    fn add(self, rhs: f64x4) -> f64x4 {
        f64x4(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
    }
}
impl AddAssign for f64x4 {
    fn add_assign(&mut self, rhs: f64x4) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
        self.3 += rhs.3;
    }
}
impl fmt::Display for f64x4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    // The `f` value implements the `Write` trait, which is what the
    // write! macro is expecting. Note that this formatting ignores the
    // various flags provided to format strings.
        write!(f, "({}, {}, {}, {})", self.0, self.1, self.2, self.3)
    }
}
/*
#[repr(simd)]
#[derive(Copy,Clone)]
struct i64x4(i64, i64, i64, i64);
*/

/*
#[cfg(target_feature = "avx")]
#[inline(always)]
fn swiz_outer(v: f64x4) -> f64x4 {
    f64x4::new(v.extract(2), v.extract(3), v.extract(0), v.extract(1))
}
#[cfg(target_feature = "avx")]
#[inline(always)]
fn swiz_inner(v: f64x4) -> f64x4 {
    f64x4::new(v.extract(1), v.extract(0), v.extract(3), v.extract(2))
}

#[cfg(target_feature = "avx")]
#[inline(always)]
fn swap_outer(v: f64x4, u:f64x4) -> (f64x4, f64x4) {
    (f64x4::new(v.extract(0), v.extract(1), u.extract(2), u.extract(3)),
    f64x4::new(u.extract(0), u.extract(1), v.extract(2), v.extract(3)))
}
#[cfg(target_feature = "avx")]
#[inline(always)]
fn swap_inner(v: f64x4, u:f64x4) -> (f64x4, f64x4) {
    (f64x4::new(v.extract(0), u.extract(1), v.extract(2), u.extract(3)),
    f64x4::new(u.extract(0), v.extract(1), u.extract(2), v.extract(3)))
}*/
/*
#[cfg(target_feature = "avx")]
impl<K: Unsigned>
    GemmNode<f64, Hierarch<f64, U4, K, U1, U4>,
                  Hierarch<f64, K, U12, U12, U1>,
                  Hierarch<f64, U4, U12, U12, U1>> for
    Ukernel<f64, Hierarch<f64, U4, K, U1, U4>,
                 Hierarch<f64, K, U12, U12, U1>,
                 Hierarch<f64, U4, U12, U12, U1>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut Hierarch<f64, U4, K, U1, U4>,
                   b: &mut Hierarch<f64, K, U12, U12, U1>,
                   c: &mut Hierarch<f64, U4, U12, U12, U1>,
                   _thr: &ThreadInfo<f64> ) -> () {
        let ap = a.get_mut_buffer() as *mut f64x4;
        let bp = b.get_mut_buffer() as *mut f64x4;
        let cp = c.get_mut_buffer() as *mut f64x4;

        let mut cA0 = f64x4::zero();
        let mut cA1 = cA0;
        let mut cA2 = cA0;
        let mut cA3 = cA0;

        let mut cB0 = cA0; 
        let mut cB1 = cA0;
        let mut cB2 = cA0;
        let mut cB3 = cA0;

        let mut cC0 = cA0;
        let mut cC1 = cA0;
        let mut cC2 = cA0;
        let mut cC3 = cA0;

        for iter in 0..a.width(){
            //breakpoint();
/*            let biA = bp[iter];
            let biB = bp[iter+4];
            let biC = bp[iter+8];*/
/*            let biA = x86_mm256_load_pd(bp.offset((12*iter) as isize));
            let biB = x86_mm256_load_pd(bp.offset((12*iter + 4) as isize));
            let biC = x86_mm256_load_pd(bp.offset((12*iter + 8) as isize));*/
            let biA = ptr::read(bp.offset((3*iter) as isize));
            let biB = ptr::read(bp.offset((3*iter + 1) as isize));
            let biC = ptr::read(bp.offset((3*iter + 2) as isize));

           // let ai0123 = x86_mm256_load_pd(ap.offset(iter as isize ));
            let ai0123 = ptr::read(ap.offset(iter as isize ));
/*            f64x4::fma(ai0123, biA, &mut cA0);
            f64x4::fma(ai0123, biB, &mut cB0);
            f64x4::fma(ai0123, biC, &mut cC0);*/
            cA0 += ai0123 * biA;
            cB0 += ai0123 * biB;
            cC0 += ai0123 * biC;

            let ai1032 = f64x4::swiz_inner(ai0123);  
            /*f64x4::fma(ai1032, biA, &mut cA1);
            f64x4::fma(ai1032, biB, &mut cB1);
            f64x4::fma(ai1032, biC, &mut cC1);*/
            cA1 += ai1032 * biA;
            cB1 += ai1032 * biB;
            cC1 += ai1032 * biC;

            let ai2301 = f64x4::swiz_outer(ai0123);  
            /*f64x4::fma(ai2301, biA, &mut cA2);
            f64x4::fma(ai2301, biB, &mut cB2);
            f64x4::fma(ai2301, biC, &mut cC2);*/
            cA2 += ai2301 * biA;
            cB2 += ai2301 * biB;
            cC2 += ai2301 * biC;

            let ai3210 = f64x4::swiz_inner(ai2301);
            /*f64x4::fma(ai3210, biA, &mut cA3);
            f64x4::fma(ai3210, biB, &mut cB3);
            f64x4::fma(ai3210, biC, &mut cC3);*/
            cA3 += ai3210 * biA;
            cB3 += ai3210 * biB;
            cC3 += ai3210 * biC;
        }

        let (cA0b,cA2b) = f64x4::blend_outer( cA0, cA2 );  
        let (cA1b,cA3b) = f64x4::blend_outer( cA1, cA3 );
        let (cA0c,cA1c) = f64x4::blend_inner( cA0b, cA1b );  
        let (cA2c,cA3c) = f64x4::blend_inner( cA2b, cA3b );  
        let mut c0 = ptr::read(cp.offset(0 as isize));
        let mut c1 = ptr::read(cp.offset(3 as isize));
        let mut c2 = ptr::read(cp.offset(6 as isize));
        let mut c3 = ptr::read(cp.offset(9 as isize));
        c0 += cA0c;
        c1 += cA1c;
        c2 += cA2c;
        c3 += cA3c;
        ptr::write(cp.offset(0 as isize), c0);
        ptr::write(cp.offset(3 as isize), c1);
        ptr::write(cp.offset(6 as isize), c2);
        ptr::write(cp.offset(9 as isize), c3);


        let (cB0b,cB2b) = f64x4::blend_outer( cB0, cB2 );  
        let (cB1b,cB3b) = f64x4::blend_outer( cB1, cB3 );
        let (cB0c,cB1c) = f64x4::blend_inner( cB0b, cB1b );  
        let (cB2c,cB3c) = f64x4::blend_inner( cB2b, cB3b );  
        let mut c4 = ptr::read(cp.offset((0+1) as isize));
        let mut c5 = ptr::read(cp.offset((3+1) as isize));
        let mut c6 = ptr::read(cp.offset((6+1) as isize));
        let mut c7 = ptr::read(cp.offset((9+1) as isize));
        c4 += cB0c;
        c5 += cB1c;
        c6 += cB2c;
        c7 += cB3c;
        ptr::write(cp.offset((0+1) as isize), c4);
        ptr::write(cp.offset((3+1) as isize), c5);
        ptr::write(cp.offset((6+1) as isize), c6);
        ptr::write(cp.offset((9+1) as isize), c7);

        let (cC0b,cC2b) = f64x4::blend_outer( cC0, cC2 );  
        let (cC1b,cC3b) = f64x4::blend_outer( cC1, cC3 );
        let (cC0c,cC1c) = f64x4::blend_inner( cC0b, cC1b );  
        let (cC2c,cC3c) = f64x4::blend_inner( cC2b, cC3b );  
        let mut c8 = ptr::read(cp.offset((0+2) as isize));
        let mut c9 = ptr::read(cp.offset((3+2) as isize));
        let mut c10 = ptr::read(cp.offset((6+2) as isize));
        let mut c11 = ptr::read(cp.offset((9+2) as isize));
        c8 += cC0c;
        c9 += cC1c;
        c10 += cC2c;
        c11 += cC3c;
        ptr::write(cp.offset((0+2) as isize), c8);
        ptr::write(cp.offset((3+2) as isize), c9);
        ptr::write(cp.offset((6+2) as isize), c10);
        ptr::write(cp.offset((9+2) as isize), c11);
    }
}*/

#[cfg(target_feature = "avx")]
impl<K: Unsigned>
    GemmNode<f64, Hierarch<f64, U4, K, U1, U4>,
                  Hierarch<f64, K, U12, U12, U1>,
                  Hierarch<f64, U4, U12, U12, U1>> for
    Ukernel<f64, Hierarch<f64, U4, K, U1, U4>,
                 Hierarch<f64, K, U12, U12, U1>,
                 Hierarch<f64, U4, U12, U12, U1>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut Hierarch<f64, U4, K, U1, U4>,
                   b: &mut Hierarch<f64, K, U12, U12, U1>,
                   c: &mut Hierarch<f64, U4, U12, U12, U1>,
                   _thr: &ThreadInfo<f64> ) -> () {
        let ap = a.get_mut_buffer() as *mut f64x4;
        let bp = b.get_mut_buffer() as *mut f64x4;
        let cp = c.get_mut_buffer() as *mut f64x4;

        let mut cA0 = f64x4::zero();
        let mut cA1 = cA0;
        let mut cA2 = cA0;
        let mut cA3 = cA0;

        let mut cB0 = cA0; 
        let mut cB1 = cA0;
        let mut cB2 = cA0;
        let mut cB3 = cA0;

        let mut cC0 = cA0;
        let mut cC1 = cA0;
        let mut cC2 = cA0;
        let mut cC3 = cA0;

        for iter in 0..a.width(){
            //breakpoint();
/*            let biA = bp[iter];
            let biB = bp[iter+4];
            let biC = bp[iter+8];*/
/*            let biA = x86_mm256_load_pd(bp.offset((12*iter) as isize));
            let biB = x86_mm256_load_pd(bp.offset((12*iter + 4) as isize));
            let biC = x86_mm256_load_pd(bp.offset((12*iter + 8) as isize));*/
            let biA = ptr::read(bp.offset((3*iter) as isize));
            let biB = ptr::read(bp.offset((3*iter + 1) as isize));
            let biC = ptr::read(bp.offset((3*iter + 2) as isize));

           // let ai0123 = x86_mm256_load_pd(ap.offset(iter as isize ));
            let ai0123 = ptr::read(ap.offset(iter as isize ));
/*            f64x4::fma(ai0123, biA, &mut cA0);
            f64x4::fma(ai0123, biB, &mut cB0);
            f64x4::fma(ai0123, biC, &mut cC0);*/
            cA0 += ai0123 * biA;
            cB0 += ai0123 * biB;
            cC0 += ai0123 * biC;

            let ai1032 = f64x4::swiz_inner(ai0123);  
            /*f64x4::fma(ai1032, biA, &mut cA1);
            f64x4::fma(ai1032, biB, &mut cB1);
            f64x4::fma(ai1032, biC, &mut cC1);*/
            cA1 += ai1032 * biA;
            cB1 += ai1032 * biB;
            cC1 += ai1032 * biC;

            let ai2301 = f64x4::swiz_outer(ai0123);  
            /*f64x4::fma(ai2301, biA, &mut cA2);
            f64x4::fma(ai2301, biB, &mut cB2);
            f64x4::fma(ai2301, biC, &mut cC2);*/
            cA2 += ai2301 * biA;
            cB2 += ai2301 * biB;
            cC2 += ai2301 * biC;

            let ai3210 = f64x4::swiz_inner(ai2301);
            /*f64x4::fma(ai3210, biA, &mut cA3);
            f64x4::fma(ai3210, biB, &mut cB3);
            f64x4::fma(ai3210, biC, &mut cC3);*/
            cA3 += ai3210 * biA;
            cB3 += ai3210 * biB;
            cC3 += ai3210 * biC;
        }

        let (cA0b,cA2b) = f64x4::blend_outer( cA0, cA2 );  
        let (cA1b,cA3b) = f64x4::blend_outer( cA1, cA3 );
        let (cA0c,cA1c) = f64x4::blend_inner( cA0b, cA1b );  
        let (cA2c,cA3c) = f64x4::blend_inner( cA2b, cA3b );  
        let mut c0 = ptr::read(cp.offset(0 as isize));
        let mut c1 = ptr::read(cp.offset(3 as isize));
        let mut c2 = ptr::read(cp.offset(6 as isize));
        let mut c3 = ptr::read(cp.offset(9 as isize));
        c0 += cA0c;
        c1 += cA1c;
        c2 += cA2c;
        c3 += cA3c;
        ptr::write(cp.offset(0 as isize), c0);
        ptr::write(cp.offset(3 as isize), c1);
        ptr::write(cp.offset(6 as isize), c2);
        ptr::write(cp.offset(9 as isize), c3);


        let (cB0b,cB2b) = f64x4::blend_outer( cB0, cB2 );  
        let (cB1b,cB3b) = f64x4::blend_outer( cB1, cB3 );
        let (cB0c,cB1c) = f64x4::blend_inner( cB0b, cB1b );  
        let (cB2c,cB3c) = f64x4::blend_inner( cB2b, cB3b );  
        let mut c4 = ptr::read(cp.offset((0+1) as isize));
        let mut c5 = ptr::read(cp.offset((3+1) as isize));
        let mut c6 = ptr::read(cp.offset((6+1) as isize));
        let mut c7 = ptr::read(cp.offset((9+1) as isize));
        c4 += cB0c;
        c5 += cB1c;
        c6 += cB2c;
        c7 += cB3c;
        ptr::write(cp.offset((0+1) as isize), c4);
        ptr::write(cp.offset((3+1) as isize), c5);
        ptr::write(cp.offset((6+1) as isize), c6);
        ptr::write(cp.offset((9+1) as isize), c7);

        let (cC0b,cC2b) = f64x4::blend_outer( cC0, cC2 );  
        let (cC1b,cC3b) = f64x4::blend_outer( cC1, cC3 );
        let (cC0c,cC1c) = f64x4::blend_inner( cC0b, cC1b );  
        let (cC2c,cC3c) = f64x4::blend_inner( cC2b, cC3b );  
        let mut c8 = ptr::read(cp.offset((0+2) as isize));
        let mut c9 = ptr::read(cp.offset((3+2) as isize));
        let mut c10 = ptr::read(cp.offset((6+2) as isize));
        let mut c11 = ptr::read(cp.offset((9+2) as isize));
        c8 += cC0c;
        c9 += cC1c;
        c10 += cC2c;
        c11 += cC3c;
        ptr::write(cp.offset((0+2) as isize), c8);
        ptr::write(cp.offset((3+2) as isize), c9);
        ptr::write(cp.offset((6+2) as isize), c10);
        ptr::write(cp.offset((9+2) as isize), c11);
    }
}*/

impl<K: Unsigned>
    GemmNode<f64, Hierarch<f64, U4, K, U1, U4>,
                  Hierarch<f64, K, U12, U12, U1>,
                  Hierarch<f64, U4, U12, U12, U1>> for
    Ukernel<f64, Hierarch<f64, U4, K, U1, U4>,
                 Hierarch<f64, K, U12, U12, U1>,
                 Hierarch<f64, U4, U12, U12, U1>> {
    #[inline(always)]
    unsafe fn run( &mut self, 
                   a: &mut Hierarch<f64, U4, K, U1, U4>,
                   b: &mut Hierarch<f64, K, U12, U12, U1>,
                   c: &mut Hierarch<f64, U4, U12, U12, U1>,
                   _thr: &ThreadInfo<f64> ) -> () {
        let ap = a.get_mut_buffer();
        let bp = b.get_mut_buffer();
        let cp = c.get_mut_buffer();

        let mut a_ind : usize = 0;
        let mut b_ind : usize = 0;
        
        asm!("vzeroall");
        asm!("
            prefetcht0 0*8($0)
            prefetcht0 8*8($0)
            prefetcht0 16*8($0)
            prefetcht0 24*8($0)
            prefetcht0 32*8($0)
            prefetcht0 40*8($0)
            "
            : : "r"(cp));
        //ymm12: a
        //ymm13: b1
        //ymm14: b2
        //ymm15: b3
        for iter in 0..a.width(){
            asm!("
                vmovapd ymm13, [$3 + 8*$1 + 0]
                vmovapd ymm14, [$3 + 8*$1 + 32]
                vmovapd ymm15, [$3 + 8*$1 + 64]

                vmovapd ymm12, [$2 + 8*$0 + 0]
                vfmadd231pd ymm0, ymm12, ymm13
                vfmadd231pd ymm1, ymm12, ymm14
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
                : : "r"(a_ind),"r"(b_ind),"r"(ap),"r"(bp) : "ymm12 ymm13 ymm14 ymm15" :"intel");
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
        : : "r"(cp) : "memory" :"intel");
    }
}
