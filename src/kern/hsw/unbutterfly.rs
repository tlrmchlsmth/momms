use matrix::{Scalar,Mat};

pub struct Unbutterfly { }
impl Unbutterfly {
    pub fn run<T: Scalar, Ct: Mat<T>>(c: &mut Ct) {
        let wpad = {(c.width()-1) / 4 + 1};
        let hpad = {(c.height()-1) / 4 + 1};

        let mut fix_one_block = |ir: usize, jr: usize| {
            let (c00, c01, c02, c03) = (c.get(0+ir,0+jr), c.get(1+ir,1+jr), c.get(3+ir,2+jr), c.get(2+ir,3+jr));
            let (c10, c11, c12, c13) = (c.get(1+ir,0+jr), c.get(0+ir,1+jr), c.get(2+ir,2+jr), c.get(3+ir,3+jr));
            let (c20, c21, c22, c23) = (c.get(3+ir,0+jr), c.get(2+ir,1+jr), c.get(0+ir,2+jr), c.get(1+ir,3+jr));
            let (c30, c31, c32, c33) = (c.get(2+ir,0+jr), c.get(3+ir,1+jr), c.get(1+ir,2+jr), c.get(0+ir,3+jr));
            c.set(0+ir,0+jr, c00); c.set(0+ir,1+jr, c01); c.set(0+ir,2+jr, c02); c.set(0+ir,3+jr, c03);
            c.set(1+ir,0+jr, c10); c.set(1+ir,1+jr, c11); c.set(1+ir,2+jr, c12); c.set(1+ir,3+jr, c13);
            c.set(2+ir,0+jr, c20); c.set(2+ir,1+jr, c21); c.set(2+ir,2+jr, c22); c.set(2+ir,3+jr, c23);
            c.set(3+ir,0+jr, c30); c.set(3+ir,1+jr, c31); c.set(3+ir,2+jr, c32); c.set(3+ir,3+jr, c33);
        };
        for jr in 0..wpad {
            for ir in 0..hpad {
                fix_one_block(4*ir,4*jr);
            }
        }
    }
}

pub struct Unbutterfly2 { }
impl Unbutterfly2 {
    pub fn run<T: Scalar, Ct: Mat<T>>(c: &mut Ct) {
        let wpad = {(c.width()-1) / 4 + 1};
        let hpad = {(c.height()-1) / 4 + 1};

        let mut fix_one_block = |ir: usize, jr: usize| {
            let (c00, c01, c02, c03) = (c.get(0+ir,0+jr), c.get(1+ir,1+jr), c.get(3+ir,2+jr), c.get(2+ir,3+jr));
            let (c10, c11, c12, c13) = (c.get(1+ir,0+jr), c.get(0+ir,1+jr), c.get(2+ir,2+jr), c.get(3+ir,3+jr));
            let (c20, c21, c22, c23) = (c.get(3+ir,0+jr), c.get(2+ir,1+jr), c.get(0+ir,2+jr), c.get(1+ir,3+jr));
            let (c30, c31, c32, c33) = (c.get(2+ir,0+jr), c.get(3+ir,1+jr), c.get(1+ir,2+jr), c.get(0+ir,3+jr));

            c.set(0+ir,0+jr, c00); c.set(1+ir,0+jr, c01); c.set(2+ir,0+jr, c02); c.set(3+ir,0+jr, c03);
            c.set(0+ir,1+jr, c10); c.set(1+ir,1+jr, c11); c.set(2+ir,1+jr, c12); c.set(3+ir,1+jr, c13);
            c.set(0+ir,2+jr, c20); c.set(1+ir,2+jr, c21); c.set(2+ir,2+jr, c22); c.set(3+ir,2+jr, c23);
            c.set(0+ir,3+jr, c30); c.set(1+ir,3+jr, c31); c.set(2+ir,3+jr, c32); c.set(3+ir,3+jr, c33);
        };
        for jr in 0..wpad {
            for ir in 0..hpad {
                fix_one_block(4*ir,4*jr);
            }
        }
    }
}
