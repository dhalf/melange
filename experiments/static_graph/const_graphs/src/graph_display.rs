use std::cell::RefCell;

use crate::{Label, Op};

pub struct DisplayLanes(RefCell<Vec<(Label, usize)>>);

impl DisplayLanes {
    pub fn new() -> Self {
        DisplayLanes(RefCell::new(Vec::new()))
    }
    
    pub fn get_lane<'a>(&'a self, lane: Label) -> DisplayLaneRef<'a> {
        let mut r = DisplayLaneRef { lanes: &self, lane, index: 0, count: 0 };
        r.init();
        r
    }

    pub fn new_lane<'a>(&'a self) -> DisplayLaneRef<'a> {
        self.0.borrow_mut().push((Label::null(), 0));
        DisplayLaneRef { lanes: &self, lane: Label::null(), index: self.len() - 1, count: 0 }
    }

    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }
}

pub struct DisplayLaneRef<'a> {
    lanes: &'a DisplayLanes,
    lane: Label,
    index: usize,
    count: usize,
}

impl<'a> DisplayLaneRef<'a> {
    fn init(&mut self) {
        let lanes = self.lanes.0.borrow();
        self.index = lanes.iter().position(|(x, _)| *x == self.lane).expect(&format!("Malformed graph: could not find referenced operation {}", self.lane));
        self.count = lanes[self.index].1;
    }

    fn refresh(&mut self) {
        let lanes = self.lanes.0.borrow();
        if self.index >= lanes.len() || self.lane != lanes[self.index].0 {
            self.init()
        }
    }
    
    fn decrement_count(&mut self) {
        assert!(self.count >= 1, "Malformed graph: referencing an operation that hasn't any registered children");
        self.count -= 1;
        self.lanes.0.borrow_mut()[self.index].1 = self.count;
    }

    fn dupplicate(&self) {
        let mut lanes = self.lanes.0.borrow_mut();
        let x = lanes[self.index];
        lanes.insert(self.index, x);
    }

    fn replace_with(&self, op: &Op) {
        self.lanes.0.borrow_mut()[self.index] = (op.label, op.children.ptr);
    }

    fn remove(self) {
        self.lanes.0.borrow_mut().remove(self.index);
    }

    pub fn star(&mut self, op: &Op) {
        self.refresh();
        println!("{}{}{} {}", "| ".repeat(self.index), "* ", "| ".repeat(self.lanes.len() - self.index - 1), op);
        self.replace_with(op);
    }

    pub fn diverge(&mut self) -> Self {
        self.refresh();
        if self.count > 1 {
            println!("{}{}{}", "| ".repeat(self.index), "|\\ ", "\\ ".repeat(self.lanes.len() - self.index - 1));
            self.decrement_count();
            self.dupplicate();
        }
        DisplayLaneRef { lanes: self.lanes, lane: self.lane, index: self.index + 1, count: self.count - 1 }
    }

    pub fn merge(&mut self, mut other: Self) -> Self {
        self.refresh();
        let a = self.index.min(other.index);
        let b = self.index.max(other.index);
        match b - a {
            0 => panic!("Malformed graph: cannot merge an operation with itself"),
            1 => {
                if other.count == 1 {
                    println!("{}|/ {}", "| ".repeat(a), "/ ".repeat(self.lanes.len() - b - 1));
                    other.remove();
                } else {
                    println!("{}|/| {}", "| ".repeat(a), "| ".repeat(self.lanes.len() - b - 1));
                    other.decrement_count();
                }
            }
            _ => {
                if other.count == 1 {
                    println!("{}|/ {}", "| ".repeat(b - 1), "/ ".repeat(self.lanes.len() - b - 1));
                    println!("{}|/|{} {}", "| ".repeat(a), "_|".repeat(b - a - 2), "| ".repeat(self.lanes.len() - b - 1));
                    other.remove();
                } else {
                    println!("{}|/| {}", "| ".repeat(b - 1), "| ".repeat(self.lanes.len() - b - 1));
                    println!("{}|/|{} {}", "| ".repeat(a), "_|".repeat(b - a - 2), "| ".repeat(self.lanes.len() - b));
                    other.decrement_count();
                }
            }
        }

        DisplayLaneRef { lanes: self.lanes, lane: self.lane, index: self.index, count: self.count }
    }
}
