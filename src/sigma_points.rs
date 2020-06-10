use generic_array::{
    sequence::Concat,
    typenum::{Sum, U1},
    ArrayLength, GenericArray,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DimName, MatrixN, VectorN};
use std::ops::Add;

pub fn van_der_merwe<DimX, DimX2, DimZ>(
    alpha: f32,
    beta: f32,
    kappa: f32,
    n: f32,
) -> (
    Box<
        dyn Fn(
            &VectorN<f32, DimX2>,
            &MatrixN<f32, DimX2>,
        ) -> GenericArray<VectorN<f32, DimX2>, Sum<Sum<U1, DimX>, DimX>>,
    >,
    GenericArray<f32, Sum<Sum<U1, DimX>, DimX>>,
    GenericArray<f32, Sum<Sum<U1, DimX>, DimX>>,
)
where
    DimZ: Dim,
    DimX: ArrayLength<VectorN<f32, DimX2>> + Add<DimX> + Add<U1> + Add<Sum<DimX, U1>>,
    DimX2: Dim + DimName,
    Sum<Sum<U1, DimX>, DimX>: ArrayLength<VectorN<f32, DimX2>> + ArrayLength<f32> + ArrayLength<()>,
    Sum<U1, DimX>: ArrayLength<VectorN<f32, DimX2>> + Add<DimX>,
    U1: ArrayLength<VectorN<f32, DimX2>> + Add<DimX>,
    DefaultAllocator: Allocator<f32, DimX2, DimX2> + Allocator<f32, DimX2> + Allocator<f32, DimZ>,
{
    let lambda_plus_n = alpha * alpha * (kappa + n);
    let lambda = lambda_plus_n - n;
    let weight = 1.0 / (2.0 * lambda_plus_n);
    let mut w_m: GenericArray<f32, Sum<Sum<U1, DimX>, DimX>> =
        GenericArray::<(), Sum<Sum<U1, DimX>, DimX>>::default()
            .iter()
            .map(|_| weight)
            .collect();
    w_m[0] = lambda / lambda_plus_n;
    let mut w_c = w_m.clone();
    w_c[0] += 1.0 - alpha * alpha + beta;
    (
        Box::new(move |x, p| {
            let c = (p * lambda_plus_n).clone().cholesky().unwrap().unpack();
            let m: GenericArray<VectorN<f32, DimX2>, DimX> =
                c.column_iter().map(|col| x - col).collect();
            let p: GenericArray<VectorN<f32, DimX2>, DimX> =
                c.column_iter().map(|col| x + col).collect();
            let z: GenericArray<VectorN<f32, DimX2>, U1> = [x.clone()].into();
            z.concat(p).concat(m)
        }),
        w_m,
        w_c,
    )
}
