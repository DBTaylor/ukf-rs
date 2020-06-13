#[cfg(test)]
mod tests {
    use nalgebra::{MatrixN, VectorN, U2, U3, MatrixMN};

    use crate::filter::Filter;
    use crate::sigma_points::van_der_merwe;

    #[test]
    fn filter_test() {
        let (s, w_m, w_c) = van_der_merwe::<typenum::U3, U3, U2>(0.5, 2.0, 0.0, 3.0);
        let x = VectorN::<f64, U3>::new(-1.0, 9.0, 0.0);
        let p = MatrixN::<f64, U3>::new(0.1, 0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01, 1.0);
        let r =  MatrixN::<f64, U2>::new(0.01, 0.0001, 0.0001, 0.01);
        let q = MatrixN::<f64, U3>::new(0.01, 0.005, 0.005, 0.005, 0.01, 0.005, 0.005, 0.005, 0.01);
        let mut filter = Filter::new(x, p,
            //f
            Box::new(|x| {
                let x = VectorN::<f64, U3>::new(x[0] + 0.1 * x[1] + 0.1 * x[2], x[1], x[2]);
                x
            }),
            s,
            w_m,
            w_c,
            r,
            //h
            Box::new(|x| VectorN::<f64, U2>::new(x[0], x[1])),
            q
        );

        let mut k_filter = kf::Filter::new(x, p,
            MatrixN::<f64, U3>::new(1.0, 0.1, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            r,
            MatrixMN::<f64, U2, U3>::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            q
        );
        let mut measurement = VectorN::<f64, U2>::new(0.0, 9.0);
        for i in 0..10000 {
            measurement[0] = i as f64;
            let (x, p) = filter.run(measurement.clone()).unwrap();
            let (x2, p2) = k_filter.run(measurement).unwrap();
            println!("{}, {}, {}, {}", (x[0] - x2[0]), (x[1] - x2[1]) / x[1], (x[2] - x2[2]) / x[2], (p[0] - p2[0]) / p[0]);
        }
    }
}