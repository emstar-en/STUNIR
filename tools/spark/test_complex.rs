#[derive(Debug)]
#[allow(dead_code)]
fn generic_func<T>(item: T) -> T {
    item
}

pub fn multiline(
    a: i32,
    b: i32,
    c: i32
) -> i32 {
    a + b + c
}

fn with_generics(map: HashMap<String, Vec<i32>>) -> bool {
    true
}
