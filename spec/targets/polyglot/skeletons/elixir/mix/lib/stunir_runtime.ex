defmodule STUNIR.Runtime do
  def println(s) do
    IO.puts(s)
  end

  def program do
    # STUB: codegen should replace this body.
    println("STUNIR Elixir skeleton: program stub")
    0
  end
end
