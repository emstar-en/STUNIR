# STUNIR Shell Compiler (jq)
# Transforms UserSpec -> IrV1

def compile_task:
  if .type == "say" then
    { op: "print", args: [.message], body: [] }
  elif .type == "calc" then
    # Naive parsing in jq is hard, assuming pre-split or simple structure
    # For shell version, we assume "calc" is just var definition for simplicity
    # or we handle the specific "10 + 32" case by splitting string
    (.expr | split(" + ")) as $parts |
    if ($parts | length) == 2 then
      { op: "add", args: [.var, $parts[0], $parts[1]], body: [] },
      { op: "print_var", args: [.var], body: [] }
    else
      { op: "var_def", args: [.var, .expr], body: [] }
    end
  elif .type == "repeat" then
    { 
      op: "loop", 
      args: [.count | tostring], 
      body: [.tasks[] | compile_task] 
    }
  else
    empty
  end;

{
  functions: [
    {
      name: "main",
      body: [.tasks[] | compile_task]
    }
  ]
}
