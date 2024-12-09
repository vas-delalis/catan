use num::{PrimInt, ToPrimitive};
use std::fmt::Debug;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// One of `u64` or `u128`.
pub trait BitboardInt:
    PrimInt + BitAndAssign + BitOrAssign + ToPrimitive + Debug + Default
{
}

impl BitboardInt for u64 {}
impl BitboardInt for u128 {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Bitboard<T: BitboardInt> {
    pub value: T,
}

impl<T: BitboardInt> Bitboard<T> {
    pub fn new(value: T) -> Self {
        Bitboard { value }
    }

    pub fn zeros() -> Self {
        Bitboard::default()
    }

    pub fn ones() -> Self {
        !Bitboard::default()
    }

    pub fn contains(&self, id: usize) -> bool {
        !(self.value & (T::one() << id)).is_zero()
    }

    pub fn add(&mut self, id: usize) {
        self.value |= T::one() << id;
    }

    pub fn remove(&mut self, id: usize) {
        self.value &= !(T::one() << id);
    }

    pub fn count_ones(&self) -> u32 {
        self.value.count_ones()
    }
}

impl<T: BitboardInt> From<T> for Bitboard<T> {
    fn from(value: T) -> Self {
        Bitboard { value }
    }
}

impl<T: BitboardInt> Iterator for Bitboard<T> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.value == T::zero() {
            return None;
        }

        // Find the first (least significant) set bit...
        let idx = self.value.trailing_zeros();
        // ... then get rid of it
        self.value &= self.value.saturating_sub(T::one());

        Some(idx as usize)
    }
}

// Ops

impl<T: BitboardInt> BitAnd for Bitboard<T> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        (self.value & rhs.value).into()
    }
}

impl<T: BitboardInt> BitOr for Bitboard<T> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        (self.value | rhs.value).into()
    }
}

impl<T: BitboardInt> BitAndAssign for Bitboard<T> {
    fn bitand_assign(&mut self, rhs: Self) {
        self.value = self.value & rhs.value;
    }
}

impl<T: BitboardInt> BitOrAssign for Bitboard<T> {
    fn bitor_assign(&mut self, rhs: Self) {
        self.value = self.value | rhs.value;
    }
}

impl<T: BitboardInt> Not for Bitboard<T> {
    type Output = Self;

    fn not(self) -> Self::Output {
        (!self.value).into()
    }
}
