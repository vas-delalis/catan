/// A multiset of `Resource` with efficient operations.
#[derive(Default)]
pub struct Bundle {
    data: u64,
}

impl Bundle {
    pub fn from_bytes(arr: [u8; 8]) -> Self {
        Bundle {
            data: u64::from_le_bytes(arr),
        }
    }

    pub fn to_bytes(&self) -> [u8; 8] {
        self.data.to_le_bytes()
    }

    pub fn contains(&self, other: Self) -> bool {
        todo!() // _mm_cmpgt_pi8
    }

    pub fn add(&self, other: Self) -> Bundle {
        todo!()
    }

    pub fn subtract(&self, other: Self) -> Bundle {
        todo!()
    }
}

impl std::fmt::Debug for Bundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_bytes())
    }
}
