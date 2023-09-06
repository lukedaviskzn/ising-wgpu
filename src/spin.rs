use std::{ops::{Mul, Neg}, fmt::Display};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spin {
    Up, // z: +1
    Down, // z: -1
}

impl From<i32> for Spin {
    fn from(value: i32) -> Self {
        if value > 0 {
            Spin::Up
        } else {
            Spin::Down
        }
    }
}

impl Into<i32> for Spin {
    fn into(self) -> i32 {
        match self {
            Spin::Up => 1,
            Spin::Down => -1,
        }
    }
}

impl Mul for Spin {
    type Output = i32;

    fn mul(self, rhs: Self) -> Self::Output {
        Into::<i32>::into(self) * Into::<i32>::into(rhs)
    }
}

impl Neg for Spin {
    type Output = Spin;

    fn neg(self) -> Self::Output {
        match self {
            Spin::Up => Spin::Down,
            Spin::Down => Spin::Up,
        }
    }
}

impl Display for Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Spin::Up => f.write_str("up"),
            Spin::Down => f.write_str("down"),
        }
    }
}
