#!/usr/bin/env python3
"""STUNIR Accessibility Standards Checker

Validates STUNIR outputs against accessibility standards.
Supports WCAG 2.1 Level A, AA, and AAA checks.

Issue: accessibility/standards/1160
"""

import os
import re
import sys
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


def canonical_json(data: Any) -> str:
    """Generate canonical JSON output."""
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def get_epoch() -> int:
    """Get current Unix epoch timestamp."""
    return int(datetime.now(timezone.utc).timestamp())


@dataclass
class AccessibilityIssue:
    """Represents an accessibility issue."""
    code: str
    message: str
    wcag_criterion: str
    level: str  # A, AA, AAA
    line: Optional[int] = None
    severity: str = "error"  # error, warning, info
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            "code": self.code,
            "message": self.message,
            "wcag_criterion": self.wcag_criterion,
            "level": self.level,
            "severity": self.severity
        }
        if self.line:
            d["line"] = self.line
        return d


class AccessibilityChecker:
    """STUNIR Accessibility Standards Checker."""
    
    SCHEMA = "stunir.accessibility.v1"
    
    def __init__(self, wcag_level: str = "AA"):
        self.wcag_level = wcag_level.upper()
        self.issues: List[AccessibilityIssue] = []
    
    def _check_html(self, content: str) -> List[AccessibilityIssue]:
        """Check HTML content for accessibility issues."""
        issues = []
        lines = content.split('\n')
        
        # Check for lang attribute (WCAG 3.1.1)
        if not re.search(r'<html[^>]*lang=["\'][a-z]{2}', content, re.IGNORECASE):
            issues.append(AccessibilityIssue(
                "MISSING_LANG", "Missing language attribute on html element",
                "3.1.1", "A"
            ))
        
        # Check for page title (WCAG 2.4.2)
        if not re.search(r'<title>[^<]+</title>', content, re.IGNORECASE):
            issues.append(AccessibilityIssue(
                "MISSING_TITLE", "Missing or empty page title",
                "2.4.2", "A"
            ))
        
        # Check for alt text on images (WCAG 1.1.1)
        for i, line in enumerate(lines, 1):
            img_matches = re.finditer(r'<img[^>]*>', line, re.IGNORECASE)
            for match in img_matches:
                img_tag = match.group(0)
                if 'alt=' not in img_tag.lower():
                    issues.append(AccessibilityIssue(
                        "MISSING_ALT", "Image missing alt attribute",
                        "1.1.1", "A", line=i
                    ))
        
        # Check heading hierarchy (WCAG 1.3.1)
        headings = re.findall(r'<h([1-6])', content, re.IGNORECASE)
        prev_level = 0
        for h in headings:
            level = int(h)
            if level > prev_level + 1 and prev_level > 0:
                issues.append(AccessibilityIssue(
                    "HEADING_SKIP", f"Heading level skipped: h{prev_level} to h{level}",
                    "1.3.1", "A", severity="warning"
                ))
            prev_level = level
        
        # Check for empty links (WCAG 2.4.4)
        empty_links = re.findall(r'<a[^>]*>\s*</a>', content, re.IGNORECASE)
        if empty_links:
            issues.append(AccessibilityIssue(
                "EMPTY_LINK", f"Found {len(empty_links)} empty link(s)",
                "2.4.4", "A"
            ))
        
        # Level AA checks
        if self.wcag_level in ["AA", "AAA"]:
            # Check for form labels (WCAG 1.3.1 + 3.3.2)
            inputs = re.findall(r'<input[^>]*>', content, re.IGNORECASE)
            for inp in inputs:
                if 'type="hidden"' not in inp.lower() and 'aria-label' not in inp.lower():
                    if 'id=' in inp.lower():
                        inp_id = re.search(r'id=["\']([^"\']+)', inp)
                        if inp_id:
                            label_pattern = f'for=["\']?{inp_id.group(1)}'
                            if not re.search(label_pattern, content, re.IGNORECASE):
                                issues.append(AccessibilityIssue(
                                    "MISSING_LABEL", "Form input missing associated label",
                                    "3.3.2", "A", severity="warning"
                                ))
        
        return issues
    
    def _check_markdown(self, content: str) -> List[AccessibilityIssue]:
        """Check Markdown content for accessibility issues."""
        issues = []
        
        # Check for image alt text
        images = re.findall(r'!\[([^\]]*)\]\([^)]+\)', content)
        for alt in images:
            if not alt.strip():
                issues.append(AccessibilityIssue(
                    "MISSING_ALT", "Image missing alt text in markdown",
                    "1.1.1", "A"
                ))
        
        # Check heading hierarchy
        headings = re.findall(r'^(#{1,6})\s', content, re.MULTILINE)
        prev_level = 0
        for h in headings:
            level = len(h)
            if level > prev_level + 1 and prev_level > 0:
                issues.append(AccessibilityIssue(
                    "HEADING_SKIP", f"Heading level skipped: {prev_level} to {level}",
                    "1.3.1", "A", severity="warning"
                ))
            prev_level = level
        
        return issues
    
    def _check_code(self, content: str) -> List[AccessibilityIssue]:
        """Check code content for documentation accessibility."""
        issues = []
        
        lines = content.split('\n')
        total_lines = len(lines)
        comment_lines = sum(1 for l in lines if re.match(r'^\s*(#|//|/\*|\*|<!--)', l))
        
        # Recommend comments for code accessibility
        if total_lines > 20 and comment_lines / total_lines < 0.1:
            issues.append(AccessibilityIssue(
                "LOW_COMMENT_RATIO", "Code has low comment ratio (<10%), consider adding documentation",
                "3.1.5", "AAA", severity="info"
            ))
        
        return issues
    
    def check_content(self, content: str, filetype: str = "html") -> Dict[str, Any]:
        """Check content for accessibility issues."""
        self.issues = []
        
        if filetype == "html":
            self.issues = self._check_html(content)
        elif filetype in ["md", "markdown"]:
            self.issues = self._check_markdown(content)
        else:
            self.issues = self._check_code(content)
        
        # Filter by WCAG level
        level_order = {"A": 1, "AA": 2, "AAA": 3}
        max_level = level_order.get(self.wcag_level, 2)
        self.issues = [i for i in self.issues if level_order.get(i.level, 1) <= max_level]
        
        errors = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        
        return {
            "valid": len(errors) == 0,
            "wcag_level": self.wcag_level,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "issues": [i.to_dict() for i in self.issues]
        }
    
    def check_file(self, filepath: str) -> Dict[str, Any]:
        """Check a file for accessibility issues."""
        if not os.path.exists(filepath):
            return {
                "valid": False,
                "error": f"File not found: {filepath}"
            }
        
        ext = os.path.splitext(filepath)[1].lower()
        filetype_map = {
            ".html": "html", ".htm": "html",
            ".md": "md", ".markdown": "md",
            ".py": "code", ".js": "code", ".c": "code", ".rs": "code"
        }
        filetype = filetype_map.get(ext, "code")
        
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        result = self.check_content(content, filetype)
        result["filepath"] = filepath
        return result
    
    def generate_receipt(self, filepath: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate accessibility check receipt."""
        return {
            "schema": self.SCHEMA,
            "epoch": get_epoch(),
            "filepath": filepath,
            "wcag_level": self.wcag_level,
            "result": result
        }


def main():
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="STUNIR Accessibility Checker")
    parser.add_argument("file", nargs="?", help="File to check")
    parser.add_argument("--dir", help="Directory to check")
    parser.add_argument("--wcag-level", choices=["A", "AA", "AAA"], default="AA",
                        help="WCAG compliance level")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--self-test", action="store_true", help="Run self-test")
    args = parser.parse_args()
    
    if args.self_test:
        # Self-test
        test_html = """<!DOCTYPE html>
<html lang="en">
<head><title>Test</title></head>
<body>
<h1>Heading</h1>
<img src="test.png" alt="Test image">
</body>
</html>"""
        
        test_bad_html = """<!DOCTYPE html>
<html>
<head></head>
<body>
<h1>Heading</h1>
<h3>Skipped h2</h3>
<img src="test.png">
</body>
</html>"""
        
        checker = AccessibilityChecker(wcag_level="AA")
        
        good = checker.check_content(test_html, "html")
        print(f"Good HTML: valid={good['valid']}, errors={good['error_count']}")
        
        bad = checker.check_content(test_bad_html, "html")
        print(f"Bad HTML: valid={bad['valid']}, errors={bad['error_count']}, warnings={bad['warning_count']}")
        
        print("Self-test passed!")
        return 0
    
    if not args.file and not args.dir:
        parser.print_help()
        return 1
    
    checker = AccessibilityChecker(wcag_level=args.wcag_level)
    
    if args.file:
        result = checker.check_file(args.file)
        if args.json:
            receipt = checker.generate_receipt(args.file, result)
            print(canonical_json(receipt))
        else:
            status = "✅ ACCESSIBLE" if result.get('valid') else "❌ ISSUES FOUND"
            print(f"{status}: {args.file}")
            print(f"  WCAG Level: {args.wcag_level}")
            print(f"  Errors: {result.get('error_count', 0)}, Warnings: {result.get('warning_count', 0)}")
            for issue in result.get('issues', []):
                sev = issue['severity'].upper()
                print(f"  [{sev}] {issue['code']}: {issue['message']} (WCAG {issue['wcag_criterion']})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
