package main
import "fmt"

func main_func() {
    fmt.Println("Compiling from Rust!")
    x := 10 + 32
    fmt.Println(x)
    for i := 0; i < 3; i++ {
        fmt.Println("Echo...")
    }
}

func main_func() { main_func() }
