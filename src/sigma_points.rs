use generic_array::{
    sequence::Concat,
    typenum::{Sum, U1},
    ArrayLength, GenericArray,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimName, MatrixN, VectorN};
use std::ops::Add;

pub fn van_der_merwe<DimX, DimX2, DimZ>(
    alpha: f64,
    beta: f64,
    kappa: f64,
    n: f64,
) -> (
    Box<
        dyn Fn(
            &VectorN<f64, DimX2>,
            &MatrixN<f64, DimX2>,
        ) -> Option<GenericArray<VectorN<f64, DimX2>, Sum<Sum<U1, DimX>, DimX>>>,
    >,
    GenericArray<f64, Sum<Sum<U1, DimX>, DimX>>,
    GenericArray<f64, Sum<Sum<U1, DimX>, DimX>>,
)
where
    DimZ: Dim,
    DimX: ArrayLength<VectorN<f64, DimX2>> + Add<DimX> + Add<U1> + Add<Sum<DimX, U1>>,
    DimX2: Dim + DimName,
    Sum<Sum<U1, DimX>, DimX>: ArrayLength<VectorN<f64, DimX2>> + ArrayLength<f64> + ArrayLength<()>,
    Sum<U1, DimX>: ArrayLength<VectorN<f64, DimX2>> + Add<DimX>,
    U1: ArrayLength<VectorN<f64, DimX2>> + Add<DimX>,
    DefaultAllocator: Allocator<f64, DimX2, DimX2> + Allocator<f64, DimX2> + Allocator<f64, DimZ>,
{
    let lambda_plus_n = alpha * alpha * (kappa + n);
    let lambda = lambda_plus_n - n;
    let weight = 1.0 / (2.0 * lambda_plus_n);
    let mut w_m: GenericArray<f64, Sum<Sum<U1, DimX>, DimX>> =
        GenericArray::<(), Sum<Sum<U1, DimX>, DimX>>::default()
            .iter()
            .map(|_| weight)
            .collect();
    w_m[0] = lambda / lambda_plus_n;
    let mut w_c = w_m.clone();
    w_c[0] += 1.0 - alpha * alpha + beta;
    (
        Box::new(move |x, p| {
            let c = match (p * lambda_plus_n).clone().cholesky() {
                Some(c) => c.unpack(),
                None => return None
            };
            let m: GenericArray<VectorN<f64, DimX2>, DimX> =
                c.column_iter().map(|col| x - col).collect();
            let p: GenericArray<VectorN<f64, DimX2>, DimX> =
                c.column_iter().map(|col| x + col).collect();
            let z: GenericArray<VectorN<f64, DimX2>, U1> = [x.clone()].into();
            Some(z.concat(p).concat(m))
        }),
        w_m,
        w_c,
    )
}
