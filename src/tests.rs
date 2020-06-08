#[cfg(test)]
mod tests {
    use nalgebra::{VectorN, MatrixN, MatrixMN, U0, U1, U2};

    use crate::filter::Filter;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn filter_test(){
        let mut filter = Filter::new(
            VectorN::<f32, U2>::new(-1.0, 9.0),//x
            MatrixN::<f32, U2>::new(//p
                0.1, 0.1,
                0.1, 0.1
            ),
            MatrixN::<f32, U2>::new(//f
                1.0, 0.1,
                0.0, 1.0
            ),
            MatrixN::<f32, U2>::new(//r
                0.1, 0.1,
                0.1, 1.0
            ),
            MatrixN::<f32, U2>::new(//h
                1.0, 0.0,
                0.0, 1.0
            ),
            MatrixN::<f32, U2>::new(//q
                1.0, 1.0,
                1.0, 1.0
            ),
        );
        let mut measurement = VectorN::<f32, U2>::new(0.0, 9.0);
        for i in 0..100{
            measurement[0] = i as f32;
            let(x, p) = filter.run(measurement);
            println!("{}, {}", x[0], x[1])
        }
    }
}

