fn main_func() {
    println!("Compiling from Rust!");
    let x = 10 + 32;
    println!("{}", x);
    for _ in 0..3 {
        println!("Echo...");
    }
}

fn main_func() { main_func(); }
