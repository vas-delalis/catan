use std::{
    cmp::Ordering,
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Index, IndexMut, Sub, SubAssign},
    simd::{
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdUint,
        u8x8,
    },
    sync::LazyLock,
};

use enum_map::{enum_map, Enum, EnumMap};

use crate::common::*;

pub static BUY_COSTS: LazyLock<EnumMap<Purchasable, Bundle>> = LazyLock::new(|| {
    enum_map! {
       Purchasable::Road => Bundle::from_slice(&[1, 0, 1, 0, 0]),
       Purchasable::Settlement => Bundle::from_slice(&[1, 1, 1, 0, 1]),
       Purchasable::City => Bundle::from_slice(&[0, 2, 0, 3, 0]),
       Purchasable::DevCard => Bundle::from_slice(&[0, 1, 0, 1, 1])
    }
});

/// An [Enum] -> [u8] map equipped with efficient operations.
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Bundle {
    pub data: u8x8,
}

impl Bundle {
    pub fn new(data: u8x8) -> Self {
        Bundle { data }
    }

    pub fn splat(value: u8) -> Self {
        Bundle {
            data: u8x8::splat(value),
        }
    }

    pub fn from_slice(src: &[u8]) -> Self {
        src.into()
    }

    pub fn reduce_sum(&self) -> u8 {
        self.data.reduce_sum()
    }

    pub fn reduce_and(&self) -> u8 {
        self.data.reduce_and()
    }

    pub fn count_nonzero(&self) -> u32 {
        // _mm_movemask_pi8 is missing from std::arch.
        // Maybe submit a patch?
        self.data.simd_gt(u8x8::splat(0)).to_bitmask().count_ones()
    }

    pub fn display<T: Enum + Display>(&self) -> String {
        let mut items = Vec::with_capacity(5);
        for i in 0..T::LENGTH {
            if self.data[i] == 0 {
                continue;
            }
            items.push(format!("{} {}", self.data[i], T::from_usize(i)));
        }

        format!("Bundle[{}]", items.join(", "))
    }
}

// impl Index<usize> for Bundle {
//     type Output = u8;

//     fn index(&self, index: usize) -> &Self::Output {
//         &self.data[index]
//     }
// }

// impl IndexMut<usize> for Bundle {
//     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
//         &mut self.data[index]
//     }
// }

// Inexplicably, `Enum` has a custom `into_usize` method instead of just implying `From<usize>`.

impl<T: Enum> Index<T> for Bundle {
    type Output = u8;

    fn index(&self, index: T) -> &Self::Output {
        &self.data[index.into_usize()]
    }
}

impl<T: Enum> IndexMut<T> for Bundle {
    fn index_mut(&mut self, index: T) -> &mut Self::Output {
        &mut self.data[index.into_usize()]
    }
}

impl Sum for Bundle {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|totals, bundle| totals + bundle)
            .unwrap_or_default()
    }
}

impl Add for Bundle {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Bundle {
            data: self.data.saturating_add(rhs.data),
        }
    }
}

impl Sub for Bundle {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Bundle {
            data: self.data.saturating_sub(rhs.data),
        }
    }
}

impl AddAssign for Bundle {
    fn add_assign(&mut self, rhs: Self) {
        self.data = self.data.saturating_add(rhs.data);
    }
}

impl SubAssign for Bundle {
    fn sub_assign(&mut self, rhs: Self) {
        self.data = self.data.saturating_sub(rhs.data);
    }
}

impl From<&[u8]> for Bundle {
    fn from(value: &[u8]) -> Self {
        Bundle {
            data: u8x8::load_or_default(value),
        }
    }
}

impl PartialOrd for Bundle {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.data.simd_lt(other.data).all() {
            Some(Ordering::Less)
        } else if self.data.simd_gt(other.data).all() {
            Some(Ordering::Greater)
        } else if self.data.simd_eq(other.data).all() {
            Some(Ordering::Equal)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_nonzero() {
        let mut b = Bundle::splat(0);
        b[0] = 5;
        assert_eq!(b.count_nonzero(), 1);
    }
}
