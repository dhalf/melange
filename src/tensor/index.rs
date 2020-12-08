//! Defines a runtime shape that can be used in conjuction with dynamic
//! type-level shapes or to express coordiantes in tensor.
//!
//! The current implementation of [`Index<U>`](Index) only wraps a
//! [`Vec<usize>`](Vec) and imposes the constraint that its length be `U`
//! where `U` is a type-level integer (see [`typenum`]).
//! 
//! [`typenum`]: https://docs.rs/typenum/1.12.0/typenum/index.html

use std::convert::TryFrom;
use std::io::{Error, ErrorKind};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use typenum::Unsigned;
use std::fmt;

#[derive(Clone)]
/// Runtime shape of length `U`. See module-level documentation.
pub struct Index<U> {
    axes: Vec<usize>,
    _phantoms: PhantomData<U>,
}

impl<U> Deref for Index<U> {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        &self.axes
    }
}

impl<U> DerefMut for Index<U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.axes
    }
}

impl<U> TryFrom<Vec<usize>> for Index<U>
where
    U: Unsigned,
{
    type Error = Error;
    fn try_from(value: Vec<usize>) -> Result<Self, Self::Error> {
        if value.len() == U::USIZE {
            Ok(Index {
                axes: value,
                _phantoms: PhantomData,
            })
        } else {
            Err(Error::new(
                ErrorKind::InvalidData,
                format!(
                    "Expected a vector of length {}, got {}.",
                    U::USIZE,
                    value.len()
                ),
            ))
        }
    }
}

impl<U> From<Index<U>> for Vec<usize> {
    fn from(index: Index<U>) -> Self {
        index.axes
    }
}

impl<U> fmt::Debug for Index<U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut tuple_output = f.debug_tuple("Index");
        for ax in self.axes.iter() {
            tuple_output.field(ax);
        }
        tuple_output.finish()
    }
}

impl<U> PartialEq for Index<U> {
    fn eq(&self, other: &Index<U>) -> bool {
        self.axes == other.axes
    }
}
