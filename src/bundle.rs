use std::{
    ops::{Add, Index, IndexMut, Sub},
    simd::{num::SimdUint, Simd},
};

use enum_map::Enum;

use crate::{Resource, RESOURCES};

/// A multiset of `Resource` with efficient operations.
#[derive(Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bundle {
    data: Simd<u8, 8>,
}

impl Bundle {
    pub fn from_array(src: [u8; 5]) -> Self {
        let mut dst = [0u8; 8];
        dst[..5].copy_from_slice(&src);
        Bundle {
            data: Simd::from_array(dst),
        }
    }
}

impl Index<Resource> for Bundle {
    type Output = u8;

    fn index(&self, index: Resource) -> &Self::Output {
        &self.data[index as usize]
    }
}

impl IndexMut<Resource> for Bundle {
    fn index_mut(&mut self, index: Resource) -> &mut Self::Output {
        &mut self.data[index as usize]
    }
}

impl Add for Bundle {
    type Output = Bundle;
    fn add(self, rhs: Self) -> Self::Output {
        Bundle {
            data: self.data.saturating_add(rhs.data),
        }
    }
}

impl Sub for Bundle {
    type Output = Bundle;
    fn sub(self, rhs: Self) -> Self::Output {
        Bundle {
            data: self.data.saturating_sub(rhs.data),
        }
    }
}

impl std::fmt::Debug for Bundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut items = Vec::with_capacity(5);
        for i in 0..Resource::LENGTH {
            if self.data[i] == 0 {
                continue;
            }
            items.push(format!("{} {:?}", self.data[i], RESOURCES[i]));
        }
        write!(f, "Bundle[{}]", items.join(", "))?;
        Ok(())
    }
}

// pub const BRICK: Bundle = Bundle { data: 1 };
// pub const GRAIN: Bundle = Bundle { data: 1 << 8 };
// pub const LUMBER: Bundle = Bundle { data: 1 << 16 };
// pub const ORE: Bundle = Bundle { data: 1 << 24 };
// pub const WOOL: Bundle = Bundle { data: 1 << 32 };
