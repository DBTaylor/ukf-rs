
use nalgebra::{VectorN, DimName, MatrixMN, MatrixN, RealField, Dim, allocator::Allocator, DefaultAllocator};


pub struct Filter<T, DimZ, DimX, DimS>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DimS: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX, DimX>
    + Allocator<T, DimX>
    + Allocator<T, DimZ, DimZ>
    + Allocator<T, DimZ, DimX>
    + Allocator<T, DimX, DimZ>
    + Allocator<T, DimZ>
{
    ///state estimate vector
    x: VectorN<T, DimX>,
    ///state covariance matrix
    p: MatrixN<T, DimX>,
    ///system model
    f: Fn(VectorN<T, DimX>) -> VectorN<T, DimX>,
    /// sigma point function
    s: Fn(VectorN<T, DimX>, MatrixN<T, DimX>) -> MatrixMN<T, DimX, DimS>,
    ///measurement noise covariance matrix
    r: MatrixN<T, DimZ>,
    ///measurement function
    h: Fn(VectorN<T, DimX>) -> VectorN<T, DimZ>,
    ///process noise covariance matrix
    q: MatrixN<T, DimX>
}

impl <T, DimZ, DimX, DimS> Filter<T, DimZ, DimX, DimS>
where
    T: RealField,
    DimZ: Dim + DimName,
    DimX: Dim + DimName,
    DimS: Dim + DimName,
    DefaultAllocator: Allocator<T, DimX, DimX>
        + Allocator<T, DimX>
        + Allocator<T, DimZ, DimZ>
        + Allocator<T, DimZ, DimX>
        + Allocator<T, DimX, DimZ>
        + Allocator<T, DimZ>
{
    pub fn new(x: VectorN<T, DimX>, p: MatrixN<T, DimX>, f: MatrixN<T, DimX>, r: MatrixN<T, DimZ>, h: MatrixMN<T, DimZ, DimX>, q: MatrixN<T, DimX>) ->  Filter<T, DimZ, DimX>{
        Filter{x, p, f, r, h, q}
    }

    pub fn run(&mut self, z: VectorN<T, DimZ>) -> (VectorN<T, DimX>, MatrixN<T, DimX>){
        //predict
        let x = &self.f * &self.x;
        let p = &self.f * &self.p * &self.f.transpose() + &self.q;
        //update
        let s = &self.h * &p * &self.h.transpose() + &self.r;
        let k = &p * &self.h.transpose() * s.try_inverse().unwrap();
        let y = z - &self.h * &x;
        self.x = x + &k * y;
        self.p = &p - k * &self.h * &p;
        (self.x.clone(), self.p.clone())
    }
}