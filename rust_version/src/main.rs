use rand::Rng;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

//cargo run --release
fn main() {
    //let file_name = "../msg_mock.txt";
    //let file_name = "../shakespeare_input.txt";
    let file_name = "../test_shakes.txt";
    let lm = train_char_lm(file_name, 4);
    // println!("{:?}", lm["Firs"]);
    let msg = generate_text(&lm, 4, 1000);
    println!("{}", msg);
}

//The Language Model
type Tlm = HashMap<String, Vec<(char, i32)>>;
type Lm = HashMap<String, Vec<(char, f64)>>;

fn normalize(accum: &[(char, i32)]) -> Vec<(char, f64)> {
    //let s: f64 = accum.iter().map(|a| f64::from(a.1)).sum();
    let s: f64 = f64::from(accum.iter().fold(0, |sum, i| sum + i.1));
    accum
        .iter()
        .map(|(chr, cnt)| (*chr, f64::from(*cnt) / s))
        .collect()
}

fn train_char_lm(filename: &str, order: usize) -> Lm {
    let mut data = std::fs::read_to_string(filename).expect("could not read file");
    //data = format!("{}{}", padding("~", order), &data);
    data = padding("~", order) + &data;
    let mut tlm = Tlm::new();
    for i in 0..(data.len() - order) {
        // let history = history_by_indices(&data, i, i + order);
        let history = String::from_utf8(data.as_bytes()[i..i + order].to_vec()).expect("not valid");

        // let current = current_char(&data, i + order);
        let current = char::from(data.as_bytes()[i + order]);
        match tlm.entry(history.to_string().clone()) {
            Entry::Vacant(e) => {
                e.insert(vec![(current, 1)]);
            }
            Entry::Occupied(mut e) => {
                if !e
                    .get_mut()
                    .iter_mut()
                    .map(|(one, _)| *one)
                    .any(|x| x == current)
                {
                    e.get_mut().push((current, 0));
                }
                for i in e.get_mut().iter_mut() {
                    if i.0 == current {
                        i.1 += 1;
                    }
                }
            }
        };
    }
    let mut lm = Lm::new();
    for m in tlm.iter() {
        lm.insert(m.0.clone(), normalize(m.1));
    }
    lm
}

fn padding(pad: &str, num: usize) -> String {
    (0..num).map(|_| pad).collect::<String>()
}

/*
fn check_char_boundary(s: &str, b: usize) -> usize {
    let mut char_bound = b;
    while !s.is_char_boundary(char_bound) {
        char_bound += 1
    }
    char_bound
}

fn rewind(token: &str, order: usize) -> &str {
    if let Some((i, _)) = token.char_indices().rev().nth(order) {
        return &token[i..];
    };
    " "
}
*/

#[allow(dead_code)]
fn rewind(token: &str, order: usize) -> String {
    token
        .chars()
        .rev()
        .take(order)
        .collect::<String>()
        .chars()
        .rev()
        .collect()
}

fn generate_text(
    model: &HashMap<String, Vec<(char, f64)>>,
    order: usize,
    nletters: usize,
) -> String {
    let mut history = padding("~", order);
    let mut out: Vec<String> = vec![];
    for _ in 0..nletters {
        // println!("textHistory: {}", history);
        let c = generate_letter(&model, history.clone().as_ref(), order);
        // println!("Char: {}", c);
        history = format!("{}{}", rewind(history.clone().as_ref(), order), c);
        out.push(c)
    }
    out.join("")
}

fn generate_letter(
    model: &HashMap<String, Vec<(char, f64)>>,
    history: &str,
    order: usize,
) -> String {
    let mut rng = rand::thread_rng();
    // println!("letterHistory: {}", history);
    let hist = rewind(history, order);
    //boo: something is wrong with the indexing
    if !model.contains_key(&hist) {
        return " ".to_string();
    }
    // println!("hist lookup: {}", hist);
    let dist = &model[&hist];
    let mut x: f64 = rng.gen();
    for (c, v) in dist.iter() {
        x -= v;
        // println!("Xrand: {}", x);
        if x <= 0.0 {
            return c.to_string();
        }
    }
    " ".to_string()
}
