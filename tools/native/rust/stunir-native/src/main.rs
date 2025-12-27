use std::env;
use std::fs;

// ---------- Tiny JSON helpers (string-search / scan) ----------

fn json_escape(s: &str) -> String {
    // Enough to keep generated IR valid JSON.
    // (Not a full JSON implementation; diagnostic core only.)
    let mut out = String::with_capacity(s.len() + 8);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Control chars -> \u00XX
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _ => out.push(ch),
        }
    }
    out
}

fn extract_value(json: &str, key: &str) -> Option<String> {
    let key_pattern = format!("\"{}\":", key);
    let start = json.find(&key_pattern)?;
    let mut rest = &json[start + key_pattern.len()..];
    rest = rest.trim_start();

    if rest.starts_with('"') {
        // String value: scan for closing quote (handles simple backslash escapes)
        let bytes = rest.as_bytes();
        let mut i = 1usize;
        let mut escaped = false;
        while i < bytes.len() {
            let b = bytes[i];
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                return Some(rest[1..i].to_string());
            }
            i += 1;
        }
        None
    } else if rest.starts_with('[') {
        // Array value - find matching bracket using a depth counter, respecting strings.
        let bytes = rest.as_bytes();
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;

        for (i, &b) in bytes.iter().enumerate() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_string = false;
                }
                continue;
            } else {
                if b == b'"' {
                    in_string = true;
                    continue;
                }
                if b == b'[' {
                    depth += 1;
                } else if b == b']' {
                    depth -= 1;
                    if depth == 0 {
                        return Some(rest[0..i + 1].to_string());
                    }
                }
            }
        }
        None
    } else {
        // Number/Boolean/null-ish token
        let end = rest
            .find(|c: char| c == ',' || c == '}' || c.is_whitespace())
            .unwrap_or(rest.len());
        Some(rest[0..end].to_string())
    }
}

fn extract_array_items(json_array: &str) -> Vec<String> {
    let mut items = Vec::new();
    let s = json_array.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return items;
    }
    let inner = &s[1..s.len() - 1];

    let bytes = inner.as_bytes();
    let mut in_string = false;
    let mut escaped = false;
    let mut depth: i32 = 0;
    let mut start: usize = 0;

    for i in 0..bytes.len() {
        let b = bytes[i];

        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        } else {
            if b == b'"' {
                in_string = true;
                continue;
            }
            if b == b'{' || b == b'[' {
                depth += 1;
            } else if b == b'}' || b == b']' {
                depth -= 1;
            } else if b == b',' && depth == 0 {
                let part = inner[start..i].trim();
                if !part.is_empty() {
                    items.push(part.to_string());
                }
                start = i + 1;
            }
        }
    }

    let tail = inner[start..].trim();
    if !tail.is_empty() {
        items.push(tail.to_string());
    }

    items
}

fn unquote_json_string(s: &str) -> String {
    // Our extract_value() returns unquoted strings for object fields.
    // But array items like ["hi"] come through as "\"hi\"" from extract_array_items.
    let t = s.trim();
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        let inner = &t[1..t.len() - 1];
        let mut out = String::new();
        let mut chars = inner.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(other) => out.push(other),
                    None => {}
                }
            } else {
                out.push(c);
            }
        }
        out
    } else {
        t.to_string()
    }
}

// ---------- Compiler (spec -> IR) ----------

fn compile_task(task_json: &str, out: &mut String) {
    let type_str = extract_value(task_json, "type").unwrap_or_default();

    if type_str == "say" {
        let msg = extract_value(task_json, "message").unwrap_or_default();
        let msg = json_escape(&msg);
        out.push_str(&format!(
            "{{\"op\":\"print\",\"args\":[\"{}\"],\"body\":[]}},",
            msg
        ));
    } else if type_str == "calc" {
        let var = extract_value(task_json, "var").unwrap_or_else(|| "x".to_string());
        let expr = extract_value(task_json, "expr").unwrap_or_else(|| "0".to_string());
        let parts: Vec<&str> = expr.split_whitespace().collect();

        if parts.len() == 3 && parts[1] == "+" {
            out.push_str(&format!(
                "{{\"op\":\"add\",\"args\":[\"{}\",\"{}\",\"{}\"],\"body\":[]}},",
                json_escape(&var),
                json_escape(parts[0]),
                json_escape(parts[2])
            ));
            out.push_str(&format!(
                "{{\"op\":\"print_var\",\"args\":[\"{}\"],\"body\":[]}},",
                json_escape(&var)
            ));
        } else {
            out.push_str(&format!(
                "{{\"op\":\"var_def\",\"args\":[\"{}\",\"{}\"],\"body\":[]}},",
                json_escape(&var),
                json_escape(&expr)
            ));
        }
    } else if type_str == "repeat" {
        let count = extract_value(task_json, "count").unwrap_or_else(|| "1".to_string());
        out.push_str(&format!(
            "{{\"op\":\"loop\",\"args\":[\"{}\"],\"body\":[",
            json_escape(&count)
        ));

        if let Some(tasks_arr) = extract_value(task_json, "tasks") {
            for t in extract_array_items(&tasks_arr) {
                compile_task(&t, out);
            }
        }
        out.push_str("]},");
    }
}

fn run_compiler(in_spec: &str, out_ir: &str) {
    let content = fs::read_to_string(in_spec).unwrap_or_default();
    let mut body_json = String::new();

    if let Some(tasks_arr) = extract_value(&content, "tasks") {
        for t in extract_array_items(&tasks_arr) {
            compile_task(&t, &mut body_json);
        }
    }

    if body_json.ends_with(',') {
        body_json.pop();
    }

    let ir = format!(
        "{{\"functions\":[{{\"name\":\"main\",\"body\":[{}]}}]}}",
        body_json
    );
    fs::write(out_ir, ir).unwrap();
}

// ---------- Emitter (IR -> bash) ----------

fn emit_body(instrs_arr: &str, indent: &str, out: &mut String) {
    for instr in extract_array_items(instrs_arr) {
        let op = extract_value(&instr, "op").unwrap_or_default();
        let args_arr = extract_value(&instr, "args").unwrap_or_else(|| "[]".to_string());
        let raw_args = extract_array_items(&args_arr);
        let args: Vec<String> = raw_args.iter().map(|x| unquote_json_string(x)).collect();

        if op == "print" {
            if let Some(msg) = args.get(0) {
                out.push_str(&format!("{}echo \"{}\"\n", indent, msg));
            }
        } else if op == "print_var" {
            if let Some(var) = args.get(0) {
                out.push_str(&format!("{}echo \"${}\"\n", indent, var));
            }
        } else if op == "var_def" {
            if args.len() >= 2 {
                out.push_str(&format!("{}{}={}\n", indent, args[0], args[1]));
            }
        } else if op == "add" {
            if args.len() >= 3 {
                // IMPORTANT: var name included in the format string
                out.push_str(&format!(
                    "{}{}=$(( {} + {} ))\n",
                    indent, args[0], args[1], args[2]
                ));
            }
        } else if op == "loop" {
            if let Some(count) = args.get(0) {
                out.push_str(&format!(
                    "{}for ((i=0; i<{}; i++)); do\n",
                    indent, count
                ));
                if let Some(body) = extract_value(&instr, "body") {
                    emit_body(&body, &format!("{}    ", indent), out);
                }
                out.push_str(&format!("{}done\n", indent));
            }
        }
    }
}

fn run_emitter(in_ir: &str, out_file: &str) {
    let content = fs::read_to_string(in_ir).unwrap_or_default();
    let mut code = String::from("#!/bin/bash\nset -e\n\n");

    if let Some(funcs_arr) = extract_value(&content, "functions") {
        for func in extract_array_items(&funcs_arr) {
            let name = extract_value(&func, "name").unwrap_or_else(|| "unknown".to_string());
            code.push_str(&format!("{}() {{\n", name));
            if let Some(body) = extract_value(&func, "body") {
                emit_body(&body, "    ", &mut code);
            }
            code.push_str("}\n");
        }
    }
    code.push_str("main\n");
    fs::write(out_file, code).unwrap();
}

fn get_arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|x| x == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn usage() -> &'static str {
    "stunir-native (diagnostic)\n\
     \n\
     compile --in-spec <spec.json> --out-ir <ir.json>\n\
     emit    --in-ir <ir.json> --target bash --out-file <app.sh>\n"
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("{}", usage());
        std::process::exit(2);
    }

    match args[1].as_str() {
        "compile" => {
            let i = get_arg(&args, "--in-spec");
            let o = get_arg(&args, "--out-ir");
            if let (Some(i), Some(o)) = (i, o) {
                run_compiler(&i, &o);
            } else {
                eprintln!("{}", usage());
                std::process::exit(2);
            }
        }
        "emit" => {
            let target = get_arg(&args, "--target").unwrap_or_else(|| "bash".to_string());
            if target != "bash" {
                eprintln!(
                    "unsupported --target: {} (only bash is supported in this diagnostic core)",
                    target
                );
                std::process::exit(2);
            }

            let i = get_arg(&args, "--in-ir");
            let o = get_arg(&args, "--out-file");
            if let (Some(i), Some(o)) = (i, o) {
                run_emitter(&i, &o);
            } else {
                eprintln!("{}", usage());
                std::process::exit(2);
            }
        }
        "-h" | "--help" | "help" => {
            eprintln!("{}", usage());
        }
        other => {
            eprintln!("unknown subcommand: {}\n\n{}", other, usage());
            std::process::exit(2);
        }
    }
}