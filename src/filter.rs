use generic_array::{ArrayLength, GenericArray};
use nalgebra::{
    allocator::Allocator, DefaultAllocator, Dim, DimName, MatrixMN, MatrixN, RealField, VectorN, U1,
};

pub struct Filter<DimZ, DimX, NumS, T>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    NumS: ArrayLength<T> + ArrayLength<VectorN<T, DimX>> + ArrayLength<VectorN<T, DimZ>>,
    DefaultAllocator: Allocator<T, DimX>
        + Allocator<T, DimZ>
        + Allocator<T, DimX, DimX>
        + Allocator<T, DimZ, DimZ>
{
    ///state estimate vector
    x: VectorN<T, DimX>,
    ///state covariance matrix
    p: MatrixN<T, DimX>,
    ///system model
    f: Box<dyn Fn(&VectorN<T, DimX>) -> VectorN<T, DimX>>,
    /// sigma point function
    s: Box<dyn Fn(&VectorN<T, DimX>, &MatrixN<T, DimX>) -> GenericArray<VectorN<T, DimX>, NumS>>,
    //
    w_m: GenericArray<T, NumS>,
    //
    w_c: GenericArray<T, NumS>,
    ///measurement noise covariance matrix
    r: MatrixN<T, DimZ>,
    ///measurement function
    h: Box<dyn Fn(&VectorN<T, DimX>) -> VectorN<T, DimZ>>,
    ///process noise covariance matrix
    q: MatrixN<T, DimX>,
}

impl<DimZ, DimX, NumS, T> Filter<DimZ, DimX, NumS, T>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    NumS: ArrayLength<T> + ArrayLength<VectorN<T, DimX>> + ArrayLength<VectorN<T, DimZ>>,
    DefaultAllocator: Allocator<T, DimX>
        + Allocator<T, DimZ>
        + Allocator<T, U1, DimX>
        + Allocator<T, U1, DimZ>
        + Allocator<T, DimX, DimX>
        + Allocator<T, DimX, DimZ>
        + Allocator<T, DimZ, DimX>
        + Allocator<T, DimZ, DimZ>
{
    pub fn new(
        x: VectorN<T, DimX>,
        p: MatrixN<T, DimX>,
        f: Box<dyn Fn(&VectorN<T, DimX>) -> VectorN<T, DimX>>,
        s: Box<
            dyn Fn(&VectorN<T, DimX>, &MatrixN<T, DimX>) -> GenericArray<VectorN<T, DimX>, NumS>,
        >,
        w_m: GenericArray<T, NumS>,
        w_c: GenericArray<T, NumS>,
        r: MatrixN<T, DimZ>,
        h: Box<dyn Fn(&VectorN<T, DimX>) -> VectorN<T, DimZ>>,
        q: MatrixN<T, DimX>,
    ) -> Filter<DimZ, DimX, NumS, T> {
        Filter {
            x,
            p,
            f,
            s,
            w_m,
            w_c,
            r,
            h,
            q,
        }
    }

    pub fn run(&mut self, z: VectorN<T, DimZ>) -> (VectorN<T, DimX>, MatrixN<T, DimX>) {
        //sigma points
        let mut upsilon = (self.s)(&self.x, &self.p);
        //predict
        for point in upsilon.iter_mut() {
            *point = (self.f)(point);
        }
        let x = upsilon
            .iter()
            .zip(self.w_m.iter())
            .fold(VectorN::<T, DimX>::zeros(), |acc, (point, weight)| {
                acc + point * *weight
            });
        let p = upsilon.iter().zip(self.w_c.iter()).fold(
            MatrixN::<T, DimX>::zeros(),
            |acc, (point, weight)| {
                let r = point - &x;
                acc + &r * &r.transpose() * *weight
            },
        ) + &self.q;
        let upsilon = (self.s)(&x, &p);
        //update
        let zeta: GenericArray<VectorN<T, DimZ>, NumS> =
            upsilon.iter().map(|point| (self.h)(point)).collect();
        let mu = zeta
            .iter()
            .zip(self.w_m.iter())
            .fold(VectorN::<T, DimZ>::zeros(), |acc, (point, weight)| {
                acc + point * *weight
            });
        let y = z - &mu;
        let p_z = zeta.iter().zip(self.w_c.iter()).fold(
            MatrixN::<T, DimZ>::zeros(),
            |acc, (point, weight)| {
                let r = point - &mu;
                acc + &r * &r.transpose() * *weight
            },
        ) + &self.r;
        let k = upsilon.iter().zip(zeta.iter()).zip(self.w_c.iter()).fold(
            MatrixMN::<T, DimX, DimZ>::zeros(),
            |acc, ((point, meas), weight)| acc + (point - &x) * (meas - &mu).transpose() * *weight,
        ) * p_z.clone().try_inverse().unwrap();
        self.x = x + &k * y;
        self.p = &p - &k * p_z * &k.transpose();
        (self.x.clone(), self.p.clone())
    }
}
