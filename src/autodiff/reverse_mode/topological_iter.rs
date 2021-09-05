use super::*;

pub(super) struct TopologicalIter {
    stack: Vec<Rc<dyn BackwardStep>>,
}

impl TopologicalIter {
    pub(super) fn new(graph: Rc<dyn BackwardStep>) -> Self {
        TopologicalIter { stack: vec![graph] }
    }
}

impl Iterator for TopologicalIter {
    type Item = Rc<dyn BackwardStep>;
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.stack.pop()?;
        for p in item.parents() {
            if p.retains_grad() && Rc::strong_count(&p) <= 2 {
                self.stack.push(p);
            }
        }
        Some(item)
    }
}
