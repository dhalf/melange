use std::fmt;
use std::marker::ConstParamTy;

#[derive(Clone, Copy, ConstParamTy, PartialEq, Eq, Debug)]
pub struct Label([u8; 32]);

impl Label {
    pub const fn null() -> Label {
        Label([0; 32])
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Label {
        Label(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    pub const fn eq(self, other: Label) -> bool {
        let mut i = 0;
        while i < self.0.len() && i < other.0.len() {
            let a = self.0[i];
            let b = other.0[i];
            if a != b {
                return false;
            }
            i+=1;
        }

        true
    }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            match write!(f, "{:02x}", byte) {
                std::result::Result::Ok(_) => continue,
                err @ std::result::Result::Err(_) => return err,
            }
        }

        return std::result::Result::Ok(());
    }
}
