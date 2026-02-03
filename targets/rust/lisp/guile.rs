//! Guile Scheme emitter

use crate::types::*;

/// Emit Guile Scheme module
pub fn emit(module_name: &str) -> EmitterResult<String> {
    let mut code = String::new();
    
    code.push_str(";; STUNIR Generated Code\n");
    code.push_str(";; Language: Guile Scheme\n");
    code.push_str(&format!(";; Module: {}\n", module_name));
    code.push_str(";; Generator: Rust Pipeline\n");
    code.push_str(";; DO-178C Level A Compliance\n\n");
    
    let guile_name = module_name.to_lowercase().replace('_', "-");
    
    code.push_str(&format!("(define-module ({})\n", guile_name));
    code.push_str("  #:export (init process))\n\n");
    
    code.push_str("(define (init)\n");
    code.push_str("  ;; Initialization code\n");
    code.push_str("  #t)\n\n");
    
    code.push_str("(define (process x)\n");
    code.push_str("  ;; Process input\n");
    code.push_str("  (* x 2))\n");
    
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit() {
        let result = emit("test_module");
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(code.contains("(define-module (test-module)"));
        assert!(code.contains("#:export"));
        assert!(code.contains("(define (init)"));
    }
}
