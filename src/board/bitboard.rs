use num::{PrimInt, ToPrimitive};
use std::fmt::{Debug, LowerHex};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// One of `u64` or `u128`.
pub trait BitboardInt:
    PrimInt + BitAndAssign + BitOrAssign + ToPrimitive + Debug + Default + LowerHex
{
}

impl BitboardInt for u64 {}
impl BitboardInt for u128 {}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
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

    pub fn is_zeros(&self) -> bool {
        self.value.is_zero()
    }

    pub fn ones() -> Self {
        !Bitboard::default()
    }

    pub fn single(id: usize) -> Self {
        let mut board = Bitboard::default();
        board.add(id);
        board
    }

    pub fn from_hex(hex: &str) -> Self {
        T::from_str_radix(hex, 16)
            .ok()
            .expect("hex string should be valid")
            .into()
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

    pub fn values(&self) -> Vec<usize> {
        let popcount = self.count_ones() as usize;
        let mut value = self.value;
        let mut result = Vec::with_capacity((popcount / 4 + 1) * 4);
        let one = T::one();
        while !value.is_zero() {
            result.push(value.trailing_zeros() as usize);
            value &= value - one;
            result.push(value.trailing_zeros() as usize);
            value &= value - one;
            result.push(value.trailing_zeros() as usize);
            value &= value - one;
            result.push(value.trailing_zeros() as usize);
            value &= value - one;
        }

        result.truncate(popcount);
        result
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
        self.value &= self.value - T::one();

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

impl<T: BitboardInt> Debug for Bitboard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bitboard {:x}", self.value)
    }
}
