//! Provides trait that define viewing functions
//! on tensors. This allows for zero cost striding,
//! broadcasting, reshaping, transposition and
//! subtensors.

use super::*;
use crate::tensor::index::Index;
use crate::tensor::layout::{DynamicLayout, Layout, StaticLayout};
use crate::tensor::shape::{
    intrinsic_strides_in_place, BroadcastShape, Same, StaticShape, StridedShape, StridedShapeDyn,
    TRUE, TransposeShape, Fit,
};
use std::convert::TryFrom;
use std::marker::PhantomData;
use typenum::{IsEqual, Unsigned};

/// Zero cost extension of the tensor to match infered or
/// given static type-level output shape.
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data) that repeats data
/// on the axes that need to be extended.
///
/// Broadcasting is valid if the following requirements
/// are met for all axes in reverse order:
/// * dimensions are equal
/// * one of the dimensions is U1
/// * the axis only exist in the largest shape
///
/// If shapes cannot be broadcasted, the code will
/// just fail to compile thanks to compile time checks.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::U2;
///
/// let a: StaticTensor<i32, Shape1D<U2>> = Tensor::try_from(vec![1, 2]).unwrap();
/// let a = Broadcast::<Shape2D<U2, U2>>::broadcast(&a);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Broadcast<Sout> {
    /// Output type.
    type Output;

    /// Performs static immutable broadcasting.
    fn broadcast(self) -> Self::Output;
}

/// Zero cost extension of the tensor to match given runtime shape
/// ([`Index`](crate::tensor::index::Index)).
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data) that repeats data
/// on the axes that need to be extended.
///
/// Broadcasting is valid if the following requirements
/// are met for all axes in reverse order:
/// * dimensions are equal
/// * one of the dimensions is U1
/// * the axis only exist in the largest shape
///
/// # Panics
/// This method panics if the tensor cannot be broadcasted
/// to the given `runtime_shape`. It also panics if the given
/// `runtime_shape` and infered or given type-level `Shape`
/// are incompatible.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::U2;
///
/// let a: DynamicTensor<i32, Shape1D<Dyn>> = Tensor::try_from(vec![1, 2]).unwrap();
/// let a = BroadcastDynamic::<Shape2D<U2, Dyn>>::broadcast_dynamic(&a, Index::<U2>::try_from(vec![2, 2]).unwrap());
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 1, 2]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait BroadcastDynamic<Sout>
where
    Sout: Shape,
{
    /// Output type.
    type Output;

    /// Performs dynamic immutable broadcasting.
    fn broadcast_dynamic(self, runtime_shape: Index<Sout::Len>) -> Self::Output;
}

/// Zero cost striding of the tensor with static type-level strides.
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data) that keeps data
/// according to the given strides.
///
/// Output shape is automatically infered at compile time.
///
/// # Examples
/// ```ignore
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: StaticTensor<i32, Shape2D<U4, U4>> = Tensor::try_from(vec![
///     1,  2,  3,  4,
///     5,  6,  7,  8,
///     9,  10, 11, 12,
///     13, 14, 15, 16,
/// ]).unwrap();
/// let a = Stride::<Shape2D<U2, U2>>::stride(&a);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 3, 9, 11]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Stride<Strides> {
    /// Output type.
    type Output;

    /// Performs static immutable striding.
    fn stride(self) -> Self::Output;
}

/// Zero cost striding of the tensor with runtime strides.
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data) that keeps data
/// according to the given strides.
///
/// # Panics
/// This method panics if the given `runtime_strides` and the infered
/// or given type-level `Strides` are incompatible.
///
/// # Examples
/// ```ignore
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: DynamicTensor<i32, Shape2D<Dyn, U4>> = Tensor::try_from(vec![
///     1,  2,  3,  4,
///     5,  6,  7,  8,
///     9,  10, 11, 12,
///     13, 14, 15, 16,
/// ]).unwrap();
/// let a = StrideDynamic::<Shape2D<U2, Dyn>>::stride_dynamic(&a, Index::<U2>::try_from(vec![2, 2]));
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 3, 9, 11]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait StrideDynamic<Strides>
where
    Strides: Shape,
{
    /// Output type.
    type Output;

    /// Performs dynamic immutable striding.
    fn stride_dynamic(self, runtime_strides: Index<Strides::Len>) -> Self::Output;
}

/// Zero cost transposition of the tensor (i.e. axes order reversal).
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data) with reversed shape
/// and strides.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::U2;
///
/// let a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, -1, 0, 1]).unwrap();
/// let a = a.transpose();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 0, -1, 1]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Transpose {
    /// Output type.
    type Output;

    /// Performs immutable transposition.
    fn transpose(self) -> Self::Output;
}

/// Zero cost reshaping of a contiguous tensor to specified
/// type-level shape.
///
/// The tensor is reinterpreted as a tensor with a
/// different but compatible type-level shape. Shapes are compatible
/// if they have the same number of elements.
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data) with a different layout.
///
/// If type-level shapes are not compatible, the code will just
/// fail to compile.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: StaticTensor<i32, Shape1D<U4>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// let a = Reshape::<Shape2D<U2, U2>>::reshape(&a);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Reshape<Sout> {
    /// Output type.
    type Output;

    /// Performs static immutable reshaping.
    fn reshape(self) -> Self::Output;
}

/// Zero cost reshaping of a contiguous tensor to given runtime shape
/// ([`Index`](crate::tensor::index::Index)).
///
/// The tensor is reinterpreted as a tensor with a
/// different but compatible runtime shape. Shapes are compatible
/// if they have the same number of elements.
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data) with a different layout.
///
/// # Panics
/// This method panics if given `runtime_shape` doesn't have the
/// same number of elements as the tensor's shape. It also panics
/// if given `runtime_shape` and infered or given type-level shape
/// `Sout` are incompatible.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: DynamicTensor<i32, Shape1D<Dyn>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// let a = ReshapeDynamic::<Shape2D<Dyn, U2>>::reshape_dynamic(&a, Index::<U2>::try_from(vec![2, 2]).unwrap());
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait ReshapeDynamic<Sout>
where
    Sout: Shape,
{
    /// Output type.
    type Output;

    /// Performs dynamic immutable reshaping.
    fn reshape_dynamic(self, runtime_shape: Index<Sout::Len>) -> Self::Output;
}

/// Zero cost reinterpretation of a dynamic tensor as a static
/// tensor with specified static type-level shape.
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data).
///
/// # Panics
/// This method panics if the tensor's runtime shape and
/// the specified static type-level shape are different.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: DynamicTensor<i32, Shape2D<Dyn, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// let a = AsStatic::<Shape2D<U2, U2>>::as_static(&a);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait AsStatic<Sout> {
    /// Output type.
    type Output;

    /// Returns a static immutable view on the tensor.
    fn as_static(self) -> Self::Output;
}

/// Zero cost reinterpretation of a static tensor as dynamic
/// keeping the same type-level shape.
///
/// Outputs a view on the tensor (i.e. a tensor whose data is
/// a borrowing of another tensor's data).
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// let a = a.as_dynamic();
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait AsDynamic {
    /// Output type.
    type Output;

    /// Returns a dynamic immutable view on the tensor.
    fn as_dynamic(self) -> Self::Output;
}

/// View on a part of a static tensor.
///
/// Outputs a view on a part of a tensor i.e. a tensor whose data is
/// a borrowing of part of the data of another tensor.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U0, U1, U2, U3};
///
/// let a: StaticTensor<i32, Shape2D<U3, U3>> = Tensor::try_from(vec![
///     1, 2, 3,
///     4, 5, 6,
///     7, 8, 9
/// ]).unwrap();
/// let a = Subview::<Shape2D<U0, U1>, Shape2D<U2, U2>>::subview(&a);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![2, 3, 5, 6]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait Subview<Offset, Sout> {
    /// Output type.
    type Output;

    /// Returns a subview on the tensor.
    fn subview(self) -> Self::Output;
}

/// View on a part of a dynamic tensor.
///
/// Outputs a view on a part of a tensor i.e. a tensor whose data is
/// a borrowing of part of the data of another tensor.
///
/// # Examples
/// ```
/// use melange::prelude::*;
/// use typenum::{U2, U4};
///
/// let a: DynamicTensor<i32, Shape2D<Dyn, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// let a = AsStatic::<Shape2D<U2, U2>>::as_static(&a);
/// let b: StaticTensor<i32, Shape2D<U2, U2>> = Tensor::try_from(vec![1, 2, 3, 4]).unwrap();
/// assert_eq!(a, b);
/// ```
pub trait SubviewDynamic<Offset, Sout>
where
    Offset: Shape,
    Sout: Shape,
{
    /// Output type.
    type Output;

    /// Returns a subview on the tensor.
    fn subview_dynamic(self, runtime_offset: Index<Offset::Len>, runtime_shape: Index<Sout::Len>) -> Self::Output;
}

macro_rules! views_impl {
    ($({$($ref_:tt $lt:lifetime $($mut_:tt)?)?} $(-> $storage_out:ty $(where D: $bound:path)?)?);*) => {$(
        macro_rules! isset_or_default {
            ($t:ty) => { $t };
            () => { D };
        }
        
        
        impl<$($lt,)? Y, Z, T, S, A, D, L, Sout> Broadcast<Sout> for $($ref_ $lt $($mut_)?)? Tensor<Static, Y, Z, T, S, A, D, L>
        where
            T: 'static,
            S: StaticShape + BroadcastShape<Sout>,
            Sout: StaticShape,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<Static, Strided, Z, T, Sout, A, isset_or_default!($($storage_out)?), DynamicLayout<Sout::Len>>;
            fn broadcast(self) -> Self::Output {
                let new_shape = Sout::to_vec();
                let current_extrinsic_strides: Vec<_> = self
                    .strides()
                    .iter()
                    .zip(S::strides().into_iter())
                    .map(|(x, y)| *x / y)
                    .collect();
                let new_intrinsic_strides = Sout::strides();
                let mut new_strides = vec![0; Sout::Len::USIZE];
                for ((((z, u), v), x), y) in new_strides
                    .iter_mut()
                    .rev()
                    .zip(S::to_vec().into_iter().rev())
                    .zip(new_shape.iter().rev())
                    .zip(current_extrinsic_strides.into_iter().rev())
                    .zip(new_intrinsic_strides.iter().rev())
                {
                    if u == *v {
                        *z = x * *y;
                    }
                }

                let new_opt_chunk_size = match new_strides
                    .iter()
                    .rev()
                    .zip(new_intrinsic_strides.into_iter().rev())
                    .skip_while(|(x, y)| **x == *y)
                    .next()
                {
                    Some((_, y)) => y,
                    None => 1,
                };

                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: DynamicLayout {
                        shape: Index::try_from(new_shape).unwrap(),
                        strides: Index::try_from(new_strides).unwrap(),
                        offset: Index::try_from(vec![0; Sout::Len::USIZE]).unwrap(),
                        num_elements: Sout::NumElements::USIZE,
                        opt_chunk_size: new_opt_chunk_size,
                    },
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? Y, Z, T, S, A, D, L, Sout> BroadcastDynamic<Sout> for $($ref_ $lt $($mut_)?)? Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            T: 'static,
            S: Shape + BroadcastShape<Sout>,
            Sout: Shape,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<Dynamic, Strided, Z, T, Sout, A, isset_or_default!($($storage_out)?), DynamicLayout<Sout::Len>>;
            fn broadcast_dynamic(self, runtime_shape: Index<Sout::Len>) -> Self::Output {
                assert!(
                    Sout::runtime_compat(&runtime_shape),
                    "`runtime_shape` is incompatible with static shape `Sout`."
                );
                let current_shape = self.shape();
                assert!(
                    runtime_shape
                        .iter()
                        .rev()
                        .zip(current_shape.iter().rev())
                        .all(|(x, y)| *x == 1 || *y == 1 || *x == *y),
                    "Cannot broadcast shapes {:?} and {:?}.",
                    current_shape,
                    runtime_shape,
                );
                let current_extrinsic_strides: Vec<_> = self
                    .strides()
                    .iter()
                    .zip(self.strides().into_iter())
                    .map(|(x, y)| *x / y)
                    .collect();
                let new_intrinsic_strides = intrinsic_strides_in_place(runtime_shape.clone().into());
                let mut new_strides = vec![0; Sout::Len::USIZE];
                for ((((z, u), v), x), y) in new_strides
                    .iter_mut()
                    .rev()
                    .zip(current_shape.into_iter().rev())
                    .zip(runtime_shape.iter().rev())
                    .zip(current_extrinsic_strides.into_iter().rev())
                    .zip(new_intrinsic_strides.iter().rev())
                {
                    if *u == *v {
                        *z = x * *y;
                    }
                }
                let new_opt_chunk_size = match new_strides
                    .iter()
                    .rev()
                    .zip(new_intrinsic_strides.into_iter().rev())
                    .skip_while(|(x, y)| **x == *y)
                    .next()
                {
                    Some((_, y)) => y,
                    None => 1,
                };
                let new_num_elements = runtime_shape.iter().product();
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: DynamicLayout {
                        shape: runtime_shape,
                        strides: Index::try_from(new_strides).unwrap(),
                        offset: Index::try_from(vec![0; Sout::Len::USIZE]).unwrap(),
                        num_elements: new_num_elements,
                        opt_chunk_size: new_opt_chunk_size,
                    },
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? Y, Z, T, S, A, D, L, Strides> Stride<Strides> for $($ref_ $lt $($mut_)?)? Tensor<Static, Y, Z, T, S, A, D, L>
        where
            T: 'static,
            S: StaticShape + StridedShape<Strides>,
            Strides: StaticShape,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<Static, Strided, Z, T, <S as StridedShape<Strides>>::Output, A, isset_or_default!($($storage_out)?), DynamicLayout<<<S as StridedShape<Strides>>::Output as Shape>::Len>>;
            fn stride(self) -> Self::Output {
                let new_strides: Vec<_> = self
                    .strides()
                    .iter()
                    .zip(Strides::to_vec().into_iter())
                    .map(|(x, y)| *x * y)
                    .collect();
                let new_opt_chunk_size = match new_strides
                    .iter()
                    .rev()
                    .zip(
                        <<S as StridedShape<Strides>>::Output as StaticShape>::strides()
                            .into_iter()
                            .rev(),
                    )
                    .skip_while(|(x, y)| **x == *y)
                    .next()
                {
                    Some((_, y)) => y,
                    None => 1,
                };
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: DynamicLayout {
                        shape: Index::try_from(<<S as StridedShape<Strides>>::Output as StaticShape>::to_vec()).unwrap(),
                        strides: Index::try_from(new_strides).unwrap(),
                        offset: Index::try_from(vec![0; <<<S as StridedShape<Strides>>::Output as Shape>::Len as Unsigned>::USIZE]).unwrap(),
                        num_elements:
                            <<<S as StridedShape<Strides>>::Output as StaticShape>::NumElements as Unsigned>::USIZE,
                        opt_chunk_size: new_opt_chunk_size,
                    },
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? Y, Z, T, S, A, D, L, Strides> StrideDynamic<Strides> for $($ref_ $lt $($mut_)?)? Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            T: 'static,
            S: Shape + StridedShapeDyn<Strides>,
            Strides: Shape,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<Dynamic, Strided, Z, T, <S as StridedShapeDyn<Strides>>::Output, A, isset_or_default!($($storage_out)?), DynamicLayout<<<S as StridedShapeDyn<Strides>>::Output as Shape>::Len>>;
            fn stride_dynamic(self, runtime_strides: Index<Strides::Len>) -> Self::Output {
                assert!(
                    Strides::runtime_compat(&runtime_strides),
                    "`runtime_strides` are incompatible with static `Strides`."
                );
                let new_shape: Vec<_> = Vec::from(self.shape())
                    .into_iter()
                    .zip(runtime_strides.iter())
                    .map(|(x, y)| x / *y + (x % *y).min(1))
                    .collect();
                let new_strides: Vec<_> = self
                    .strides()
                    .iter()
                    .zip(runtime_strides.into_iter())
                    .map(|(x, y)| *x * y)
                    .collect();
                let new_opt_chunk_size = match new_strides
                    .iter()
                    .rev()
                    .zip(
                        intrinsic_strides_in_place(new_shape.clone())
                            .into_iter()
                            .rev(),
                    )
                    .skip_while(|(x, y)| **x == *y)
                    .next()
                {
                    Some((_, y)) => y,
                    None => 1,
                };
                let new_num_elements = new_shape.iter().product();
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: DynamicLayout {
                        shape: Index::try_from(new_shape).unwrap(),
                        strides: Index::try_from(new_strides).unwrap(),
                        offset: Index::try_from(vec![0; <<<S as StridedShapeDyn<Strides>>::Output as Shape>::Len as Unsigned>::USIZE]).unwrap(),
                        num_elements: new_num_elements,
                        opt_chunk_size: new_opt_chunk_size,
                    },
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? X, Y, Z, T, S, A, D, L> Transpose for $($ref_ $lt $($mut_)?)? Tensor<X, Y, Z, T, S, A, D, L>
        where
            T: 'static,
            S: Shape + TransposeShape,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<X, Strided, Transposed, T, <S as TransposeShape>::Output, A, isset_or_default!($($storage_out)?), DynamicLayout<S::Len>>;
            fn transpose(self) -> Self::Output {
                let shape: Vec<_> = Vec::from(self.shape()).into_iter().rev().collect();
                let strides: Vec<_> = Vec::from(self.strides()).into_iter().rev().collect();
                let num_elements = self.num_elements();
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: DynamicLayout {
                        shape: Index::try_from(shape).unwrap(),
                        strides: Index::try_from(strides).unwrap(),
                        offset: Index::try_from(vec![0; S::Len::USIZE]).unwrap(),
                        num_elements,
                        opt_chunk_size: 1,
                    },
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? T, S, A, D, L, Sout> Reshape<Sout> for $($ref_ $lt $($mut_)?)? Tensor<Static, Contiguous, Normal, T, S, A, D, L>
        where
            T: 'static,
            S: StaticShape,
            Sout: StaticShape,
            S::NumElements: IsEqual<Sout::NumElements>,
            <S::NumElements as IsEqual<Sout::NumElements>>::Output: TRUE,
            $($(D: $bound,)?)?
        {
            type Output = Tensor<Static, Contiguous, Normal, T, Sout, A, isset_or_default!($($storage_out)?), StaticLayout<Sout>>;
            fn reshape(self) -> Self::Output {
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: StaticLayout::new(),
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? T, S, A, D, L, Sout> ReshapeDynamic<Sout> for $($ref_ $lt $($mut_)?)? Tensor<Dynamic, Contiguous, Normal, T, S, A, D, L>
        where
            T: 'static,
            S: Shape,
            Sout: Shape,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<Dynamic, Contiguous, Normal, T, Sout, A, isset_or_default!($($storage_out)?), DynamicLayout<Sout::Len>>;
            fn reshape_dynamic(self, runtime_shape: Index<Sout::Len>) -> Self::Output {
                assert!(
                    Sout::runtime_compat(&runtime_shape),
                    "`runtime_shape` is incompatible with static shape `Sout`."
                );
                let num_elements = self.num_elements();
                let new_num_elements = runtime_shape.iter().product();
                assert_eq!(
                    num_elements,
                    new_num_elements,
                    "Cannot reshape tensor of shape {:?} into shape {:?}.",
                    num_elements,
                    new_num_elements,
                );
                let new_strides = intrinsic_strides_in_place(runtime_shape.clone().into());
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: DynamicLayout {
                        shape: runtime_shape,
                        strides: Index::try_from(new_strides).unwrap(),
                        offset: Index::try_from(vec![0; Sout::Len::USIZE]).unwrap(),
                        num_elements,
                        opt_chunk_size: num_elements,
                    },
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? Y, Z, T, S, A, D, L, Sout> AsStatic<Sout> for $($ref_ $lt $($mut_)?)? Tensor<Dynamic, Y, Z, T, S, A, D, L>
        where
            T: 'static,
            Sout: StaticShape,
            S: Same<Sout>,
            <S as Same<Sout>>::Output: TRUE,
            $($(D: $bound,)?)?
            L: Layout<Sout::Len>,
        {
            type Output = Tensor<Static, Y, Z, T, Sout, A, isset_or_default!($($storage_out)?), L>;
            fn as_static(self) -> Self::Output {
                let new_shape = Sout::to_vec();
                let current_shape = Vec::from(self.shape());
                assert_eq!(
                    new_shape, current_shape,
                    "Dynamic tensor is not compatible with static shape: got {:?} instead of {:?}",
                    new_shape, current_shape,
                );
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: self.layout.clone(),
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? Y, Z, T, S, A, D, L> AsDynamic for $($ref_ $lt $($mut_)?)? Tensor<Static, Y, Z, T, S, A, D, L>
        where
            T: 'static,
            S: Shape,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<Dynamic, Y, Z, T, S, A, isset_or_default!($($storage_out)?), L>;
            fn as_dynamic(self) -> Self::Output {
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: self.layout.clone(),
                    _phantoms: PhantomData,
                }
            }
        }

        impl<$($lt,)? Y, Z, T, S, A, D, L, Offset, Sout> Subview<Offset, Sout> for $($ref_ $lt $($mut_)?)? Tensor<Static, Y, Z, T, S, A, D, L>
        where
            Offset: StaticShape,
            Sout: StaticShape,
            S: Shape<Len = Sout::Len> + Fit<Offset, Sout>,
            <S as Fit<Offset, Sout>>::Output: TRUE,
            $($(D: $bound,)?)?
            L: Layout<S::Len>,
        {
            type Output = Tensor<Static, Strided, Z, T, Sout, A, isset_or_default!($($storage_out)?), DynamicLayout<Sout::Len>>;
            fn subview(self) -> Self::Output {
                let strides = self.strides();
                let opt_chunk_size_index = Sout::Len::USIZE - self.shape().into_iter().zip(Sout::to_vec().into_iter()).rev().take_while(|(d, d_out)| *d == d_out).count();
                let opt_chunk_size = self.opt_chunk_size().min(if opt_chunk_size_index >= 2 {
                    Sout::strides()[opt_chunk_size_index - 2]
                } else {
                    Sout::NumElements::USIZE
                });
                println!("index: {}, opt: {}", opt_chunk_size_index, opt_chunk_size);
                Tensor {
                    data: $($ref_ $($mut_)?)? self.data,
                    layout: DynamicLayout {
                        shape: Index::try_from(Sout::to_vec()).unwrap(),
                        strides: Index::try_from(strides).unwrap(),
                        offset: Index::try_from(Offset::to_vec()).unwrap(),
                        num_elements: Sout::NumElements::USIZE,
                        opt_chunk_size,
                    },
                    _phantoms: PhantomData,
                }
            }
        }
    )*};
}

views_impl! {
    {};
    { &'a } -> &'a [T] where D: Deref<Target = [T]>;
    { &'a mut } -> &'a mut [T] where D: DerefMut<Target = [T]>
}
