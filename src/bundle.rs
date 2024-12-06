use std::{
    fmt::Display,
    ops::{Add, Index, IndexMut, Sub},
    simd::{num::SimdUint, Simd},
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
#[derive(Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bundle {
    data: Simd<u8, 8>,
}

impl Bundle {
    pub fn from_slice(src: &[u8]) -> Self {
        Bundle {
            data: Simd::from_slice(src),
        }
    }

    pub fn sum(&self) -> u8 {
        self.data.reduce_sum()
    }

    pub fn reduce_and(&self) -> u8 {
        self.data.reduce_and()
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

impl From<&[u8]> for Bundle {
    fn from(value: &[u8]) -> Self {
        Bundle {
            data: Simd::load_or_default(value),
        }
    }
}

// pub const BRICK: Bundle = Bundle { data: 1 };
// pub const GRAIN: Bundle = Bundle { data: 1 << 8 };
// pub const LUMBER: Bundle = Bundle { data: 1 << 16 };
// pub const ORE: Bundle = Bundle { data: 1 << 24 };
// pub const WOOL: Bundle = Bundle { data: 1 << 32 };
