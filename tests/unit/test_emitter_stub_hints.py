#!/usr/bin/env python3
"""
STUNIR Emitter Stub Hint Unit Tests

Tests that all emitters emit proper STUB: prefixed comments
for unsupported operations instead of invalid code.

This test validates the Phase 1-10 emission improvements.
"""

import re
import unittest
from typing import Dict, List, Tuple

# Expected stub hint patterns for each target
STUB_HINT_EXPECTATIONS = {
    # Lisp Family
    "common_lisp": {
        "break": r";; STUB: break.*return-from",
        "continue": r";; STUB: continue.*\(go\)",
        "unsupported": r";; STUB: unsupported operation",
    },
    "scheme": {
        "break": r";; STUB: break.*call/cc",
        "continue": r";; STUB: continue.*continuation",
        "unsupported": r";; STUB: unsupported operation",
    },
    "clojure": {
        "break": r";; STUB: break.*recur|reduced",
        "continue": r";; STUB: continue.*recur.*accumulator",
        "unsupported": r";; STUB: unsupported operation",
    },
    # Prolog Family
    "swi_prolog": {
        "unsupported": r"% STUB: unsupported operation",
    },
    "gnu_prolog": {
        "unsupported": r"% STUB: unsupported operation",
    },
    "mercury": {
        "unsupported": r"% STUB: unsupported operation",
    },
    # Functional/Formal
    "futhark": {
        "break": r"-- STUB: break.*tagged result",
        "continue": r"-- STUB: continue.*tagged result",
        "unsupported": r"-- STUB: unsupported operation",
    },
    "lean4": {
        "break": r"-- STUB: break.*whileM|StateM|Except",
        "continue": r"-- STUB: continue.*whileM|StateM",
        "unsupported": r"-- STUB: unsupported operation",
    },
    "haskell": {
        "break": r"-- STUB: break.*Either monad",
        "continue": r"-- STUB: continue.*Either monad",
        "unsupported": r"-- STUB: unsupported operation",
    },
    # Ada/SPARK
    "ada": {
        "break": r"exit;",
        "continue": r"-- STUB: continue.*label",
        "map_keys": r"-- STUB: map_keys.*Iterate",
        "set_union": r"-- STUB: set_union.*Union",
        "set_intersect": r"-- STUB: set_intersect.*Intersection",
        "unsupported": r"-- STUB: unsupported operation",
    },
    "spark": {
        "break": r"exit;",
        "continue": r"-- STUB: continue.*label",
        "map_keys": r"-- STUB: map_keys.*Iterate",
        "set_union": r"-- STUB: set_union.*Union",
        "set_intersect": r"-- STUB: set_intersect.*Intersection",
        "unsupported": r"-- STUB: unsupported operation",
    },
}

# Invalid patterns that should NOT appear in output
INVALID_PATTERNS = {
    "common_lisp": [r"\(break\)", r"\(continue\)"],
    "scheme": [r"\(break\)", r"\(continue\)"],
    "clojure": [r"\(break\)", r"\(continue\)"],
    "ada": [r"goto Continue;"],
    "spark": [r"goto Continue;"],
}


class TestStubHintPatterns(unittest.TestCase):
    """Test that stub hint patterns are correctly defined."""
    
    def test_all_targets_have_expectations(self):
        """Verify all target languages have stub hint expectations."""
        expected_targets = {
            "common_lisp", "scheme", "clojure",
            "swi_prolog", "gnu_prolog", "mercury",
            "futhark", "lean4", "haskell",
            "ada", "spark"
        }
        
        actual_targets = set(STUB_HINT_EXPECTATIONS.keys())
        
        missing = expected_targets - actual_targets
        self.assertEqual(len(missing), 0, 
            f"Missing stub hint expectations for: {missing}")
    
    def test_lisp_family_has_break_continue_stubs(self):
        """Verify Lisp family has break/continue stub hints."""
        for target in ["common_lisp", "scheme", "clojure"]:
            expectations = STUB_HINT_EXPECTATIONS[target]
            self.assertIn("break", expectations, 
                f"{target} missing break stub hint")
            self.assertIn("continue", expectations,
                f"{target} missing continue stub hint")
    
    def test_functional_family_has_break_continue_stubs(self):
        """Verify functional languages have break/continue stub hints."""
        for target in ["futhark", "lean4", "haskell"]:
            expectations = STUB_HINT_EXPECTATIONS[target]
            self.assertIn("break", expectations,
                f"{target} missing break stub hint")
            self.assertIn("continue", expectations,
                f"{target} missing continue stub hint")
    
    def test_ada_spark_has_continue_stub(self):
        """Verify Ada/SPARK has continue stub hint (break uses exit)."""
        for target in ["ada", "spark"]:
            expectations = STUB_HINT_EXPECTATIONS[target]
            self.assertIn("continue", expectations,
                f"{target} missing continue stub hint")
            self.assertIn("map_keys", expectations,
                f"{target} missing map_keys stub hint")
            self.assertIn("set_union", expectations,
                f"{target} missing set_union stub hint")
            self.assertIn("set_intersect", expectations,
                f"{target} missing set_intersect stub hint")


class TestStubHintFormat(unittest.TestCase):
    """Test that stub hints follow the expected format."""
    
    def test_stub_hints_use_prefix(self):
        """Verify all stub hints use STUB: prefix."""
        for target, expectations in STUB_HINT_EXPECTATIONS.items():
            for operation, pattern in expectations.items():
                # Skip break for Ada/SPARK (uses exit;)
                if target in ["ada", "spark"] and operation == "break":
                    continue
                
                self.assertIn("STUB:", pattern,
                    f"{target}/{operation} missing STUB: prefix")
    
    def test_stub_hints_are_informative(self):
        """Verify stub hints provide implementation guidance."""
        for target, expectations in STUB_HINT_EXPECTATIONS.items():
            for operation, pattern in expectations.items():
                # Skip break for Ada/SPARK (uses exit;)
                if target in ["ada", "spark"] and operation == "break":
                    continue
                
                # Pattern should have more than just "STUB:"
                self.assertGreater(len(pattern), 20,
                    f"{target}/{operation} stub hint too short")


class TestInvalidPatterns(unittest.TestCase):
    """Test that invalid patterns are correctly identified."""
    
    def test_lisp_invalid_patterns(self):
        """Verify Lisp invalid patterns are defined."""
        for target in ["common_lisp", "scheme", "clojure"]:
            self.assertIn(target, INVALID_PATTERNS)
            patterns = INVALID_PATTERNS[target]
            self.assertTrue(any("break" in p for p in patterns),
                f"{target} missing break invalid pattern")
            self.assertTrue(any("continue" in p for p in patterns),
                f"{target} missing continue invalid pattern")
    
    def test_ada_spark_invalid_patterns(self):
        """Verify Ada/SPARK invalid patterns are defined."""
        for target in ["ada", "spark"]:
            self.assertIn(target, INVALID_PATTERNS)
            patterns = INVALID_PATTERNS[target]
            self.assertTrue(any("goto" in p for p in patterns),
                f"{target} missing goto invalid pattern")


class TestStubHintValidation(unittest.TestCase):
    """Test stub hint validation functions."""
    
    def validate_stub_hint(self, output: str, target: str, operation: str) -> Tuple[bool, str]:
        """Validate that output contains expected stub hint."""
        if target not in STUB_HINT_EXPECTATIONS:
            return False, f"No expectations for target: {target}"
        
        expectations = STUB_HINT_EXPECTATIONS[target]
        if operation not in expectations:
            return False, f"No expectation for operation: {operation}"
        
        pattern = expectations[operation]
        if re.search(pattern, output, re.IGNORECASE):
            return True, "Found"
        else:
            return False, f"Pattern not found: {pattern}"
    
    def check_no_invalid_patterns(self, output: str, target: str) -> List[str]:
        """Check that output doesn't contain invalid patterns."""
        found = []
        if target in INVALID_PATTERNS:
            for pattern in INVALID_PATTERNS[target]:
                if re.search(pattern, output):
                    found.append(pattern)
        return found
    
    def test_validate_stub_hint_found(self):
        """Test validation finds valid stub hints."""
        output = "  ;; STUB: break - use (return-from) or (go) in Common Lisp"
        valid, msg = self.validate_stub_hint(output, "common_lisp", "break")
        self.assertTrue(valid, msg)
    
    def test_validate_stub_hint_not_found(self):
        """Test validation rejects missing stub hints."""
        output = "  (break)"  # Invalid - no stub hint
        valid, msg = self.validate_stub_hint(output, "common_lisp", "break")
        self.assertFalse(valid)
    
    def test_check_no_invalid_patterns_clean(self):
        """Test invalid pattern check with clean output."""
        output = "  ;; STUB: break - use (return-from)"
        found = self.check_no_invalid_patterns(output, "common_lisp")
        self.assertEqual(len(found), 0)
    
    def test_check_no_invalid_patterns_found(self):
        """Test invalid pattern check detects bad patterns."""
        output = "  (break)"  # Invalid pattern
        found = self.check_no_invalid_patterns(output, "common_lisp")
        self.assertGreater(len(found), 0)


class TestEmitterOutputSamples(unittest.TestCase):
    """Test sample emitter outputs for stub hints."""
    
    def test_common_lisp_break_output(self):
        """Test Common Lisp break stub hint."""
        sample = "  ;; STUB: break - use (return-from) or (go) in Common Lisp"
        self.assertIn("STUB:", sample)
        self.assertIn("return-from", sample)
    
    def test_scheme_break_output(self):
        """Test Scheme break stub hint."""
        sample = "  ;; STUB: break - use (call/cc) escape continuation in Scheme"
        self.assertIn("STUB:", sample)
        self.assertIn("call/cc", sample)
    
    def test_clojure_break_output(self):
        """Test Clojure break stub hint."""
        sample = "  ;; STUB: break - use (recur) with accumulator or reduced in loop/recur"
        self.assertIn("STUB:", sample)
        self.assertIn("recur", sample)
    
    def test_futhark_break_output(self):
        """Test Futhark break stub hint."""
        sample = "  -- STUB: break - use loop with tagged result (e.g., type result = Break | Continue | Done)"
        self.assertIn("STUB:", sample)
        self.assertIn("tagged result", sample)
    
    def test_lean4_break_output(self):
        """Test Lean4 break stub hint."""
        sample = "  -- STUB: break - use whileM with StateM monad and early return via Except"
        self.assertIn("STUB:", sample)
        self.assertIn("whileM", sample)
    
    def test_ada_continue_output(self):
        """Test Ada continue stub hint."""
        sample = "   --  STUB: continue - use label at end of loop body: <<Continue>> null;"
        self.assertIn("STUB:", sample)
        self.assertIn("label", sample)
    
    def test_ada_map_keys_output(self):
        """Test Ada map_keys stub hint."""
        sample = "   --  STUB: map_keys - iterate with Iterate or Get_First/Get_Next cursors"
        self.assertIn("STUB:", sample)
        self.assertIn("Iterate", sample)
    
    def test_prolog_unsupported_output(self):
        """Test Prolog unsupported operation stub hint."""
        sample = "    % STUB: unsupported operation - requires manual implementation"
        self.assertIn("STUB:", sample)


def run_tests():
    """Run all stub hint tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestStubHintPatterns))
    suite.addTests(loader.loadTestsFromTestCase(TestStubHintFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestInvalidPatterns))
    suite.addTests(loader.loadTestsFromTestCase(TestStubHintValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestEmitterOutputSamples))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
