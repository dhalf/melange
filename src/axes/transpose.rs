use super::*;

pub trait Append<An> {
    type Output;
}

impl<An> Append<An> for Ax0 {
    type Output = Ax<An, Ax0>;
}

impl<An, A, As> Append<An> for Ax<A, As>
where
    As: Append<An>,
{
    type Output = Ax<A, <As as Append<An>>::Output>;
}

pub trait Transpose {
    type Output;
}

impl<A, As> Transpose for Ax<A, As>
where
    As: Transpose,
    <As as Transpose>::Output: Append<A>,
{
    type Output = <<As as Transpose>::Output as Append<A>>::Output;
}

impl Transpose for Ax0 {
    type Output = Ax0;
}
