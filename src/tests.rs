#[cfg(test)]
mod tests {
    use nalgebra::{MatrixN, VectorN, U2, U3};

    use crate::filter::Filter;
    use crate::sigma_points::van_der_merwe;

    #[test]
    fn filter_test() {
        let (s, w_m, w_c) = van_der_merwe::<typenum::U3, U3, U2>(0.5, 2.0, 0.0, 3.0);
        let mut filter = Filter::new(
            //x
            VectorN::<f32, U3>::new(-1.0, 9.0, 0.0),
            //p
            MatrixN::<f32, U3>::new(0.1, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 1.0),
            //f
            Box::new(|x| {
                let x = VectorN::<f32, U3>::new(x[0] + 0.1 * x[1] + 0.1 * x[2], x[1], x[2]);
                x
            }),
            //s
            s,
            //w_m
            w_m,
            //w_c
            w_c,
            //r
            MatrixN::<f32, U2>::new(0.01, 0.0001, 0.0001, 0.01),
            //h
            Box::new(|x| VectorN::<f32, U2>::new(x[0], x[1])),
            //q
            MatrixN::<f32, U3>::new(0.01, 0.005, 0.005, 0.005, 0.01, 0.005, 0.005, 0.005, 0.01),
        );
        let mut measurement = VectorN::<f32, U2>::new(0.0, 9.0);
        for i in 0..3 {
            measurement[0] = i as f32;
            let (x, p) = filter.run(measurement);
            println!("{}, {}, {}, {}", x[0], x[1], x[2], p[0])
        }
    }
}
