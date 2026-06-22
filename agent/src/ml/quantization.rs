use std::arch::x86_64::*;

use common::{GameState, Image};
use tch::Tensor;

use crate::{agents::Evaluator, ml::Model};

const ACTIVATION_SCALE: i64 = 64;
const WEIGHT_SCALE: i64 = 64;
const WEIGHT_SCALE_LOG2: i32 = 6;
const OUTPUT_SCALE: i64 = 64;

// #[repr(align(64))]
struct Buffer32(Vec<i32>);

// #[repr(align(64))]
struct Buffer8(Vec<i8>);

// #[repr(align(64))]
struct Outputs {
    linear0: Buffer32,
    crelu0: Buffer8,
    linear1: Buffer32,
    crelu1: Buffer8,
    output: Buffer32,
}

pub struct QuantizedEvaluator {
    linear0: LinearLayer,
    crelu0: CReLU,
    linear1: LinearLayer,
    crelu1: CReLU,
    output: LinearLayer,
}

impl QuantizedEvaluator {
    pub fn new(model: &Model) -> Self {
        let linear0 = LinearLayer::from_tensors(
            &model.var_store().variables()["layer0.weight"],
            &model.var_store().variables()["layer0.bias"],
        );
        let crelu0 = CReLU {};
        let linear1 = LinearLayer::from_tensors(
            &model.var_store().variables()["layer1.weight"],
            &model.var_store().variables()["layer1.bias"],
        );
        let crelu1 = CReLU {};
        let output = LinearLayer::from_tensors(
            &model.var_store().variables()["output.weight"],
            &model.var_store().variables()["output.bias"],
        );

        Self {
            linear0,
            crelu0,
            linear1,
            crelu1,
            output,
        }
    }
}

impl<G: GameState + Image> Evaluator<G> for QuantizedEvaluator {
    fn evaluate(&self, game_state: &G, arbiter: G::Player) -> f32 {
        let input_tensor = game_state.image(arbiter);
        let input = (&input_tensor * ACTIVATION_SCALE).internal_cast_byte(false);
        let mut input: Vec<i8> = input.try_into().unwrap();
        input.extend_from_slice(&vec![0; 32 - input.len()]);

        let mut outputs = Outputs {
            linear0: Buffer32(vec![0; self.linear0.padded_num_out.next_multiple_of(32)]),
            crelu0: Buffer8(vec![0; self.linear1.padded_num_in]),
            linear1: Buffer32(vec![0; self.linear1.padded_num_out.next_multiple_of(32)]),
            crelu1: Buffer8(vec![0; self.output.padded_num_in]),
            output: Buffer32(vec![0; self.output.padded_num_out]),
        };

        unsafe {
            self.linear0.run(&input, &mut outputs.linear0.0);
            self.crelu0.run(&outputs.linear0.0, &mut outputs.crelu0.0);

            self.linear1.run(&outputs.crelu0.0, &mut outputs.linear1.0);
            self.crelu1.run(&outputs.linear1.0, &mut outputs.crelu1.0);

            self.output.run(&outputs.crelu1.0, &mut outputs.output.0);
        };
        outputs.output.0[0] as f32 / 64.0
    }
}

struct LinearLayer {
    padded_num_in: usize,
    padded_num_out: usize,
    weights: Vec<i8>,
    bias: Vec<i32>,
    num_out_chunks: usize,
}

impl LinearLayer {
    fn from_tensors(weights: &Tensor, bias: &Tensor) -> Self {
        let num_out = weights.size()[0] as usize;
        let num_in = weights.size()[1] as usize;
        let padded_num_in = num_in.next_multiple_of(32);
        let padded_num_out = num_out.next_multiple_of(4);
        let padding_rows = padded_num_out - num_out;
        let padding_cols = padded_num_in - num_in;

        let weights: Vec<i8> = (weights * WEIGHT_SCALE)
            .pad(
                [0, padding_cols as i64, 0, padding_rows as i64],
                "constant",
                0.0,
            )
            .internal_cast_byte(false)
            .flatten(0, 1)
            .try_into()
            .unwrap();

        let bias: Vec<i32> = (bias * WEIGHT_SCALE * OUTPUT_SCALE)
            .internal_cast_int(false)
            .try_into()
            .unwrap();

        Self {
            padded_num_in,
            padded_num_out,
            num_out_chunks: padded_num_out / 4,
            weights,
            bias,
        }
    }

    unsafe fn run(&self, input: &[i8], output: &mut [i32]) {
        let register_width = 256 / 8;
        let num_in_chunks = input.len() / register_width;

        debug_assert!(input.len() % 32 == 0);
        debug_assert_ne!(self.num_out_chunks, 0);

        let input = input.as_ptr() as *const __m256i;
        let weights = (&self.weights).as_ptr() as *const __m256i;
        let bias = (&self.bias).as_ptr() as *const __m128i;
        let output = (output).as_mut_ptr() as *mut __m128i;

        for i in 0..self.num_out_chunks {
            let offset0 = (i * 4 + 0) * num_in_chunks;
            let offset1 = (i * 4 + 1) * num_in_chunks;
            let offset2 = (i * 4 + 2) * num_in_chunks;
            let offset3 = (i * 4 + 3) * num_in_chunks;

            unsafe {
                let mut sum0 = _mm256_setzero_si256();
                let mut sum1 = _mm256_setzero_si256();
                let mut sum2 = _mm256_setzero_si256();
                let mut sum3 = _mm256_setzero_si256();

                for j in 0..num_in_chunks {
                    let batch = _mm256_loadu_si256(input.add(j));

                    let w0 = _mm256_loadu_si256(weights.add(offset0 + j));
                    let w1 = _mm256_loadu_si256(weights.add(offset1 + j));
                    let w2 = _mm256_loadu_si256(weights.add(offset2 + j));
                    let w3 = _mm256_loadu_si256(weights.add(offset3 + j));
                    sum0 = _mm256_dpbusd_epi32(sum0, batch, w0);
                    sum1 = _mm256_dpbusd_epi32(sum1, batch, w1);
                    sum2 = _mm256_dpbusd_epi32(sum2, batch, w2);
                    sum3 = _mm256_dpbusd_epi32(sum3, batch, w3);
                }

                let bias = _mm_load_si128(bias.add(i));

                let outval = m256_haddx4(sum0, sum1, sum2, sum3, bias);
                let outval = _mm_srai_epi32::<WEIGHT_SCALE_LOG2>(outval);

                _mm_store_si128(output.wrapping_add(i), outval);
            }
        }
    }
}

struct CReLU {}

impl CReLU {
    unsafe fn run(&self, input: &[i32], output: &mut [i8]) {
        let out_register_width = 256 / 8;
        let num_out_chunks = input.len() / out_register_width;
        debug_assert_ne!(num_out_chunks, 0);

        let input = input.as_ptr() as *const __m256i;
        let output = (output).as_mut_ptr() as *mut __m256i;

        unsafe {
            let zero = _mm256_setzero_si256();
            let control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

            for i in 0..num_out_chunks {
                let in0 = _mm256_packus_epi32(
                    _mm256_loadu_si256(input.add(i * 4 + 0)),
                    _mm256_loadu_si256(input.add(i * 4 + 1)),
                );
                let in1 = _mm256_packus_epi32(
                    _mm256_loadu_si256(input.add(i * 4 + 2)),
                    _mm256_loadu_si256(input.add(i * 4 + 3)),
                );

                let result = _mm256_permutevar8x32_epi32(
                    _mm256_max_epi8(_mm256_packus_epi16(in0, in1), zero),
                    control,
                );

                _mm256_storeu_si256(output.add(i), result);
            }
        };
    }
}

fn m256_haddx4(
    sum0: __m256i,
    sum1: __m256i,
    sum2: __m256i,
    sum3: __m256i,
    bias: __m128i,
) -> __m128i {
    unsafe {
        let sum0 = _mm256_hadd_epi32(sum0, sum1);
        let sum2 = _mm256_hadd_epi32(sum2, sum3);

        let sum0 = _mm256_hadd_epi32(sum0, sum2);

        let sum128lo = _mm256_castsi256_si128(sum0);
        let sum128hi = _mm256_extracti128_si256::<1>(sum0);

        _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias)
    }
}
