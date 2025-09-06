pub struct Node {
    visits: u16,
    children: Vec<Option<Box<Node>>>,
    to_play: i8,
    prior: f64, // TODO: smaller? quantized?
    total_value: f64,
}

impl Node {
    pub fn value(&self) -> f64 {
        self.total_value / (self.visits as f64)
    }
}

pub struct Search {
    pb_c_base: f64,
    pb_c_init: f64,
}

impl Search {
    fn new(&self) -> Self {
        Search {
            pb_c_base: 1.0,
            pb_c_init: 1.0,
        }
    }

    fn score(&self, parent: &Node, child: &Node) -> f64 {
        let mut pb_c = (((parent.visits as f64 + self.pb_c_base + 1.0) / self.pb_c_base)
            + self.pb_c_init)
            .ln();
        pb_c *= f64::sqrt(parent.visits as f64) / (child.visits as f64 + 1.0);

        let prior_score = pb_c * child.prior;
        let value_score = child.value();
        prior_score + value_score
    }
}
