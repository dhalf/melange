//! THIS IS AN EXPERIMENTAL IMPLEMENTATION OF TYPE
//! LEVEL GRAPHS THROUGH CONST GENERICS

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(adt_const_params)]
#![allow(long_running_const_eval)]
#![feature(const_option)]

use sha2_const::Sha256;
use std::{marker::ConstParamTy, ops::Add, fmt};

const CAPACITY: usize = 128;

mod graph_display;
use graph_display::DisplayLanes;

mod label;
use label::Label;

const LABEL1: Label = Label::from_bytes(Sha256::new().update(b"foo").finalize());
const LABEL2: Label = Label::from_bytes(Sha256::new().update(b"bar").finalize());

const ROOT1: List<Op> = List::root(LABEL1);
const ROOT2: List<Op> = List::root(LABEL2);

#[derive(Clone, Copy, ConstParamTy, PartialEq, Eq, Debug)]
enum Option<T> {
    Some(T),
    None,
}
use crate::Option::{Some, None};

impl<T: Copy> Option<T> {
    const fn unwrap(self) -> T {
        match self {
            Some(t) => t,
            None => panic!("Tried to unwrap a None variant"),
        }
    }
}

macro_rules! iter_list {
    (for ($elem:ident$(; $ix:ident)?) in ($it:expr) $blk:block) => {
        let mut i = 0;
        while let Some($elem) = $it.storage[i] {
            i+=1;
            $(let $ix = i - 1;)?
            $blk
        }
    };
}

macro_rules! zip_list {
    (for ($elem1:ident, $elem2:ident) in ($it1:expr, $it2:expr) $blk:block) => {
        let mut i = 0;
        while let (Some($elem1), Some($elem2)) = ($it1.storage[i], $it2.storage[i]) {
            i+=1;
            $blk
        }
    };
}

macro_rules! map_list {
    ($it:expr, |$elem:ident| $blk:block) => {
        {
            let mut res = $it;
            let mut i = 0;
            while let Some($elem) = $it.storage[i] {
                i+=1;
                res.storage[i] = Some($blk);
            }
            
            res
        }
    };
}

struct Tensor(f32);

#[derive(Clone, Copy, ConstParamTy, PartialEq, Eq, Debug)]
enum OpCode {
    Root,
    Add,
    Exp,
}

impl OpCode {
    const fn as_bytes(self) -> &'static [u8; 4] {
        match self {
            OpCode::Root => b"root",
            OpCode::Add => b"add0",
            OpCode::Exp => b"exp0",
        }
    }
}

#[derive(Clone, Copy, ConstParamTy, PartialEq, Eq, Debug)]
struct Op {
    label: Label,
    op_code: OpCode,
    parents: (Option<Label>, Option<Label>),
    children: List<Label>,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (children: {})", self.label, self.children.ptr)
    }
}

trait Run {
    type Storage<'a>;

    fn run<'a>(self, storage: Self::Storage<'a>); 
}

struct Dispatch<const C: OpCode>;

impl Run for Dispatch<{OpCode::Add}> {
    type Storage<'a> = (&'a Tensor, &'a Tensor, &'a mut Tensor);
    
    fn run<'a>(self, storage: Self::Storage<'a>) {
        storage.2.0 = storage.0.0 + storage.1.0
    }
}

impl Run for Dispatch<{OpCode::Exp}> {
    type Storage<'a> = (&'a Tensor, &'a mut Tensor);
    
    fn run<'a>(self, storage: Self::Storage<'a>) {
        storage.1.0 = storage.0.0.exp()
    }
}

impl Op {
    const fn root(label: Label) -> Self {
        Op { label, op_code: OpCode::Root, parents: (None, None), children: List::new() }
    }

    const fn eq(self, other: Self) -> bool {
        self.label.eq(other.label)
    }

    const fn update(self, other: Self) -> Self {
        Op {
            label: self.label,
            op_code: self.op_code,
            parents: self.parents,
            children: self.children.cat(other.children),
        }
    }

    const fn register_child(self, label: Label) -> Self {
        Op {
            label: self.label,
            op_code: self.op_code,
            parents: self.parents,
            children: self.children.push(label),
        }
    }
}

#[derive(Clone, Copy, ConstParamTy, PartialEq, Eq, Debug)]
struct List<T> {
    storage: [Option<T>; CAPACITY],
    ptr: usize,
}

impl<T: Copy> List<T> {
    const fn new() -> Self {
        List { storage: [None; CAPACITY], ptr: 0 }
    }

    const fn push(mut self, elt: T) -> Self {
        assert!(self.ptr < CAPACITY);
        self.storage[self.ptr] = Some(elt);
        self.ptr += 1;
        self
    }

    const fn cat(mut self, other: Self) -> Self {
        assert!(self.ptr + other.ptr - 1 < CAPACITY);

        let mut i = 0;
        while i < other.ptr {
            self.storage[self.ptr] = other.storage[i];
            self.ptr += 1;
            i += 1;
        }
        self
    }

    const fn last(self) -> Option<T> {
        if self.ptr > 0 {
            self.storage[self.ptr - 1]
        } else {
            None
        }
    }

    const fn replace_last(mut self, elt: T) -> Self {
        if self.ptr > 0 {
            self.storage[self.ptr - 1] = Some(elt);
        }
        self
    }
}

impl List<Op> {
    const fn root(label: Label) -> Self {
        let res = List::new();
        res.push(Op::root(label))
    }

    const fn position(self, label: Label) -> Option<usize> {
        iter_list! {
            for (op; ix) in (self) {
                if op.label.eq(label) {
                    return Some(ix);
                }
            }
        };
        
        None
    }

    const fn merge(mut self, other: Self) -> Self {
        let mut tail = List::new();
        iter_list! {
            for (op) in (other) {
                match self.position(op.label) {
                    Some(i) => self.storage[i] = Some(self.storage[i].unwrap().update(op)),
                    None => tail = tail.push(op),
                }
            }
        }
        self.cat(tail)
    }

    const fn register_child(self, label: Label) -> Self {
        let a = self.last().unwrap().register_child(label);
        self.replace_last(a)
    }

    const fn unary_op(self, op_code: OpCode) -> Self {
        let parent = self.last().unwrap().label;
        let label = Label::from_bytes(Sha256::new()
            .update(op_code.as_bytes())
            .update(parent.as_bytes())
            .finalize());
        self
            .register_child(label)
            .push(Op { label, op_code, parents: (Some(parent), None), children: List::new() })
    }

    const fn binary_op(self, other: Self, op_code: OpCode) -> Self {
        let parent1 = self.last().unwrap().label;
        let parent2 = other.last().unwrap().label;
        let label = Label::from_bytes(Sha256::new()
            .update(op_code.as_bytes())
            .update(&parent1.as_bytes())
            .update(&parent2.as_bytes())
            .finalize());
        
        self
            .register_child(label)
            .merge(other.register_child(label))
            .push(Op { label, op_code, parents: (Some(parent1), Some(parent2)), children: List::new() })
    }

    fn print_graph(&self) {
        let lanes = DisplayLanes::new();

        for op in self.storage {
            if let Some(op) = op {
                match op.parents {
                    (None, None) => lanes.new_lane().star(&op),
                    (Some(parent), None) => lanes.get_lane(parent).diverge().star(&op),
                    (Some(parent1), Some(parent2)) => lanes.get_lane(parent1).diverge().merge(lanes.get_lane(parent2)).star(&op),
                    _ => panic!("Malformed graph: operation has a second parent without a first parent")
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Graph<const G: List<Op>>;

impl<const G: List<Op>> Graph<G> {
    fn print(self) {
        G.print_graph();
    }
}

const ADD_CODE: OpCode = OpCode::Add;

impl<const G: List<Op>, const H: List<Op>> Add<Graph<H>> for Graph<G>
where
    Graph<{G.binary_op(H, ADD_CODE)}>:,
{
    type Output = Graph<{G.binary_op(H, ADD_CODE)}>;
    fn add(self, _rhs: Graph<H>) -> Self::Output {
        Graph
    }
}

const EXP_CODE: OpCode = OpCode::Exp;

impl<const G: List<Op>> Graph<G> {
    fn exp(self) -> Graph<{G.unary_op(EXP_CODE)}>
    where
        Graph<{G.unary_op(EXP_CODE)}>:,
    {
        Graph
    }
}

fn main() {
    let a: Graph<ROOT1> = Graph;
    let b: Graph<ROOT2> = Graph;

    let c = a + b;
    // c.print();

    // println!("{:?}", ROOT1.last().unwrap().register_child(LABEL1));
    // println!("{:?}", ROOT1.register_child(LABEL1));

    let d = a.exp();
    // d.print();
    let e = c + d;
    e.print();
}
