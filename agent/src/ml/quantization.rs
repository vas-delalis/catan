use std::{
    alloc::{Layout, handle_alloc_error},
    arch::x86_64::*,
    ptr::slice_from_raw_parts_mut,
};

use common::{GameState, Image};
use tch::Tensor;

use crate::{agents::Evaluator, ml::Model};

const ACTIVATION_SCALE: i64 = 64;
const WEIGHT_SCALE: i64 = 64;
const WEIGHT_SCALE_LOG2: i32 = 6;
const OUTPUT_SCALE: i64 = 64;
const REGISTER_WIDTH: usize = 256 / (size_of::<i8>() * 8);

struct Outputs {
    linear0: *mut [i32],
    crelu0: *mut [i8],
    linear1: *mut [i32],
    crelu1: *mut [i8],
    output: *mut [i32],
}

impl Outputs {
    fn clear(&self) {
        unsafe {
            (self.linear0 as *mut i32).write_bytes(0, self.linear0.len());
            (self.crelu0 as *mut i8).write_bytes(0, self.crelu0.len());
            (self.linear1 as *mut i32).write_bytes(0, self.linear1.len());
            (self.crelu1 as *mut i8).write_bytes(0, self.crelu1.len());
            (self.output as *mut i32).write_bytes(0, self.output.len());
        }
    }
}

pub struct QuantizedEvaluator {
    linear0: LinearLayer,
    crelu0: CReLU,
    linear1: LinearLayer,
    crelu1: CReLU,
    output: LinearLayer,
    outputs: Outputs,
}

impl QuantizedEvaluator {
    pub fn new(model: &Model) -> Self {
        let linear0 = LinearLayer::from_tensors(
            &model.var_store().variables()["layer0.weight"],
            &model.var_store().variables()["layer0.bias"],
        );
        let crelu0 = CReLU {
            num_out_chunks: linear0.padded_num_out.next_multiple_of(32) / REGISTER_WIDTH,
        };
        let linear1 = LinearLayer::from_tensors(
            &model.var_store().variables()["layer1.weight"],
            &model.var_store().variables()["layer1.bias"],
        );
        let crelu1 = CReLU {
            num_out_chunks: linear1.padded_num_out.next_multiple_of(32) / REGISTER_WIDTH,
        };
        let output = LinearLayer::from_tensors(
            &model.var_store().variables()["output.weight"],
            &model.var_store().variables()["output.bias"],
        );

        let outputs = Outputs {
            linear0: allocate_aligned_slice(
                linear0.padded_num_out.next_multiple_of(32) * size_of::<i32>(),
            ),
            crelu0: allocate_aligned_slice(linear1.padded_num_in * size_of::<i8>()),
            linear1: allocate_aligned_slice(
                linear1.padded_num_out.next_multiple_of(32) * size_of::<i32>(),
            ),
            crelu1: allocate_aligned_slice(output.padded_num_in * size_of::<i8>()),
            output: allocate_aligned_slice(output.padded_num_out * size_of::<i32>()),
        };

        Self {
            linear0,
            crelu0,
            linear1,
            crelu1,
            output,
            outputs,
        }
    }
}

impl<G: GameState + Image> Evaluator<G> for QuantizedEvaluator {
    fn evaluate(&self, game_state: &G, arbiter: G::Player) -> f32 {
        self.outputs.clear();
        let input_tensor = game_state.image(arbiter);
        let input_tensor = (&input_tensor * ACTIVATION_SCALE).internal_cast_byte(false);

        let input: *mut i8 =
            allocate_aligned(input_tensor.numel().next_multiple_of(32) * size_of::<i8>());

        unsafe {
            input.copy_from(input_tensor.data_ptr() as *const i8, input_tensor.numel());

            self.linear0.run(input, self.outputs.linear0 as *mut i32);
            self.crelu0.run(
                self.outputs.linear0 as *const i32,
                self.outputs.crelu0 as *mut i8,
            );
            self.linear1.run(
                self.outputs.crelu0 as *const i8,
                self.outputs.linear1 as *mut i32,
            );
            self.crelu1.run(
                self.outputs.linear1 as *const i32,
                self.outputs.crelu1 as *mut i8,
            );
            self.output.run(
                self.outputs.crelu1 as *const i8,
                self.outputs.output as *mut i32,
            );
            (self.outputs.output as *mut i32).read() as f32 / WEIGHT_SCALE as f32
        }
    }
}

fn allocate_aligned<T>(size: usize) -> *mut T {
    unsafe {
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        ptr as *mut T
    }
}

fn allocate_aligned_slice<T>(size: usize) -> *mut [T] {
    unsafe {
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = std::alloc::alloc_zeroed(layout);
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        slice_from_raw_parts_mut(ptr as *mut T, size / size_of::<T>())
    }
}

struct LinearLayer {
    padded_num_in: usize,
    padded_num_out: usize,
    num_out_chunks: usize,
    weights: *const i8,
    bias: *const i32,
}

impl LinearLayer {
    fn from_tensors(weights: &Tensor, bias: &Tensor) -> Self {
        let num_out = weights.size()[0] as usize;
        let num_in = weights.size()[1] as usize;
        let padded_num_in = num_in.next_multiple_of(32);
        let padded_num_out = num_out.next_multiple_of(4);
        let padding_rows = padded_num_out - num_out;
        let padding_cols = padded_num_in - num_in;

        let weights_buffer: *mut i8 = allocate_aligned(padded_num_in * padded_num_out);
        let bias_buffer: *mut i32 = allocate_aligned(size_of::<i32>() * padded_num_out);

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

        unsafe {
            weights_buffer.copy_from(weights.as_ptr(), weights.len());
            bias_buffer.copy_from(bias.as_ptr(), bias.len());
        }

        Self {
            padded_num_in,
            padded_num_out,
            num_out_chunks: padded_num_out / 4,
            weights: weights_buffer,
            bias: bias_buffer,
        }
    }

    unsafe fn run(&self, input: *const i8, output: *mut i32) {
        let num_in_chunks = self.padded_num_in / REGISTER_WIDTH;

        debug_assert!(self.padded_num_in % 32 == 0);
        debug_assert_ne!(self.num_out_chunks, 0);

        let input = input as *const __m256i;
        let weights = self.weights as *const __m256i;
        let bias = self.bias as *const __m128i;
        let output = output as *mut __m128i;

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
                    let batch = _mm256_load_si256(input.add(j));

                    let w0 = _mm256_load_si256(weights.add(offset0 + j));
                    let w1 = _mm256_load_si256(weights.add(offset1 + j));
                    let w2 = _mm256_load_si256(weights.add(offset2 + j));
                    let w3 = _mm256_load_si256(weights.add(offset3 + j));
                    sum0 = _mm256_dpbusd_epi32(sum0, batch, w0);
                    sum1 = _mm256_dpbusd_epi32(sum1, batch, w1);
                    sum2 = _mm256_dpbusd_epi32(sum2, batch, w2);
                    sum3 = _mm256_dpbusd_epi32(sum3, batch, w3);
                }

                let bias = _mm_load_si128(bias.add(i));

                let outval = m256_haddx4(sum0, sum1, sum2, sum3, bias);
                let outval = _mm_srai_epi32::<WEIGHT_SCALE_LOG2>(outval);

                _mm_store_si128(output.add(i), outval);
            }
        }
    }
}

struct CReLU {
    num_out_chunks: usize,
}

impl CReLU {
    unsafe fn run(&self, input: *const i32, output: *mut i8) {
        debug_assert_ne!(self.num_out_chunks, 0);

        let input = input as *const __m256i;
        let output = output as *mut __m256i;

        unsafe {
            let zero = _mm256_setzero_si256();
            let control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

            for i in 0..self.num_out_chunks {
                let in0 = _mm256_packs_epi32(
                    _mm256_load_si256(input.add(i * 4 + 0)),
                    _mm256_load_si256(input.add(i * 4 + 1)),
                );
                let in1 = _mm256_packs_epi32(
                    _mm256_load_si256(input.add(i * 4 + 2)),
                    _mm256_load_si256(input.add(i * 4 + 3)),
                );

                let result = _mm256_permutevar8x32_epi32(
                    _mm256_max_epi8(_mm256_packs_epi16(in0, in1), zero),
                    control,
                );

                _mm256_store_si256(output.add(i), result);
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
