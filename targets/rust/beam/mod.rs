//! BEAM VM emitters (Erlang/Elixir)
//!
//! Supports: Erlang, Elixir bytecode and source

use crate::types::*;

/// BEAM language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BEAMLanguage {
    Erlang,
    Elixir,
    ErlangBytecode,
}

impl std::fmt::Display for BEAMLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            BEAMLanguage::Erlang => write!(f, "Erlang"),
            BEAMLanguage::Elixir => write!(f, "Elixir"),
            BEAMLanguage::ErlangBytecode => write!(f, "Erlang Bytecode"),
        }
    }
}

/// Emit BEAM code
pub fn emit(language: BEAMLanguage, module_name: &str) -> EmitterResult<String> {
    match language {
        BEAMLanguage::Erlang => emit_erlang(module_name),
        BEAMLanguage::Elixir => emit_elixir(module_name),
        BEAMLanguage::ErlangBytecode => emit_bytecode(module_name),
    }
}

fn emit_erlang(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("% STUNIR Generated Erlang\n");
    code.push_str(&format!("% Module: {}\n", module_name));
    code.push_str("% Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("-module({}).\n", module_name.to_lowercase()));
    code.push_str("-export([start/0, process/1]).\n\n");
    
    code.push_str("start() ->\n");
    code.push_str("    spawn(fun() -> process(0) end).\n\n");
    
    code.push_str("process(State) ->\n");
    code.push_str("    receive\n");
    code.push_str("        {get, From} ->\n");
    code.push_str("            From ! {state, State},\n");
    code.push_str("            process(State);\n");
    code.push_str("        {set, NewState} ->\n");
    code.push_str("            process(NewState);\n");
    code.push_str("        stop ->\n");
    code.push_str("            ok\n");
    code.push_str("    end.\n");
    
    Ok(code)
}

fn emit_elixir(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("# STUNIR Generated Elixir\n");
    code.push_str(&format!("# Module: {}\n", module_name));
    code.push_str("# Generator: Rust Pipeline\n\n");
    
    code.push_str(&format!("defmodule {} do\n", module_name));
    code.push_str("  use GenServer\n\n");
    
    code.push_str("  # Client API\n");
    code.push_str("  def start_link(initial_state \\ 0) do\n");
    code.push_str("    GenServer.start_link(__MODULE__, initial_state, name: __MODULE__)\n");
    code.push_str("  end\n\n");
    
    code.push_str("  def get_state do\n");
    code.push_str("    GenServer.call(__MODULE__, :get_state)\n");
    code.push_str("  end\n\n");
    
    code.push_str("  def set_state(new_state) do\n");
    code.push_str("    GenServer.cast(__MODULE__, {:set_state, new_state})\n");
    code.push_str("  end\n\n");
    
    code.push_str("  # Server callbacks\n");
    code.push_str("  @impl true\n");
    code.push_str("  def init(initial_state) do\n");
    code.push_str("    {:ok, initial_state}\n");
    code.push_str("  end\n\n");
    
    code.push_str("  @impl true\n");
    code.push_str("  def handle_call(:get_state, _from, state) do\n");
    code.push_str("    {:reply, state, state}\n");
    code.push_str("  end\n\n");
    
    code.push_str("  @impl true\n");
    code.push_str("  def handle_cast({:set_state, new_state}, _state) do\n");
    code.push_str("    {:noreply, new_state}\n");
    code.push_str("  end\n");
    code.push_str("end\n");
    
    Ok(code)
}

fn emit_bytecode(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str("% STUNIR Generated Erlang Bytecode (Abstract Format)\n");
    code.push_str(&format!("% Module: {}\n", module_name));
    code.push_str("% Generator: Rust Pipeline\n\n");
    
    code.push_str("{module, ");
    code.push_str(&module_name.to_lowercase());
    code.push_str("}.\n");
    code.push_str("{exports, [{start, 0}, {process, 1}]}.\n");
    code.push_str("{attributes, []}.\n");
    code.push_str("{labels, 10}.\n\n");
    
    code.push_str("{function, start, 0, 2}.\n");
    code.push_str("  {label,1}.\n");
    code.push_str("  {func_info,{atom,");
    code.push_str(&module_name.to_lowercase());
    code.push_str("},{atom,start},0}.\n");
    code.push_str("  {label,2}.\n");
    code.push_str("  return.\n");
    
    Ok(code)
}
