//! Contains tools to describe the layout of a tensor.
//!
//! Layouts come in two flavours:
//! * [`StaticLayout`] that encodes a static and contiguous layout,
//! * [`DynamicLayout`] encompasses all the remaining cases.
//!
//! Contiguous tensors data can be directly read from memory
//! since it is already laid out the right manner (row major).
//! Hence static and contiguous tensors do not need to store
//! any layout information: their shape is known at compile
//! time and their data is properly laid out.
//! For that reason, [`StaticLayout`] is actually a ZST which
//! means it has no overhead.
//!
//! As soon as a tensor is not static or not contiguous, either
//! its shape, or strides, or both have to be stored. It that
//! case [`DynamicLayout`] should be used.
//!
//! Both kinds of layouts implement the `Layout` trait which
//! defines a common interface.
//!
//! [`Layout`]: Layout
//! [`StaticLayout`]: StaticLayout
//! [`DynamicLayout`]: DynamicLayout

#[doc(hidden)]
pub mod layout;

#[doc(hidden)]
pub mod static_layout;

#[doc(hidden)]
pub mod dynamic_layout;

#[doc(inline)]
pub use layout::Layout;

#[doc(inline)]
pub use static_layout::StaticLayout;

#[doc(inline)]
pub use dynamic_layout::DynamicLayout;
