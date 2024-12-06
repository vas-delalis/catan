use num::PrimInt;
use std::fmt::Debug;
use std::ops::BitAndAssign;

pub struct BitIterator<T: PrimInt + BitAndAssign> {
    value: T,
}

impl<T: PrimInt + BitAndAssign> BitIterator<T> {
    pub fn new(bitboard: T) -> Self {
        BitIterator { value: bitboard }
    }
}

impl<T: PrimInt + BitAndAssign + Debug> Iterator for BitIterator<T> {
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
