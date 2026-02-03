#!/usr/bin/env python3
"""
STUNIR Hypothesis Strategies
============================

Custom Hypothesis strategies for generating STUNIR-specific test data.
"""

try:
    from hypothesis import strategies as st
    from hypothesis import settings, Phase
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Provide stub for when hypothesis is not installed
    class st:
        @staticmethod
        def text(*args, **kwargs): return None
        @staticmethod  
        def dictionaries(*args, **kwargs): return None
        @staticmethod
        def lists(*args, **kwargs): return None
        @staticmethod
        def integers(*args, **kwargs): return None
        @staticmethod
        def floats(*args, **kwargs): return None
        @staticmethod
        def booleans(*args, **kwargs): return None
        @staticmethod
        def none(*args, **kwargs): return None
        @staticmethod
        def one_of(*args, **kwargs): return None
        @staticmethod
        def just(*args, **kwargs): return None
        @staticmethod
        def sampled_from(*args, **kwargs): return None
        @staticmethod
        def recursive(*args, **kwargs): return None
        @staticmethod
        def fixed_dictionaries(*args, **kwargs): return None
        @staticmethod
        def binary(*args, **kwargs): return None

import string


def json_strategy():
    """Generate arbitrary JSON-like structures."""
    if not HYPOTHESIS_AVAILABLE:
        return None
    
    # Base JSON values
    json_primitives = st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-2**53, max_value=2**53),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=100),
    )
    
    # Recursive JSON structures
    return st.recursive(
        json_primitives,
        lambda children: st.one_of(
            st.lists(children, max_size=10),
            st.dictionaries(
                st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=20),
                children,
                max_size=10
            )
        ),
        max_leaves=50
    )


def ir_strategy():
    """Generate STUNIR IR-like structures."""
    if not HYPOTHESIS_AVAILABLE:
        return None
    
    # IR type names
    ir_types = st.sampled_from(["i32", "i64", "f32", "f64", "bool", "void", "ptr"])
    
    # IR identifiers
    identifier = st.text(
        alphabet=string.ascii_letters + "_",
        min_size=1,
        max_size=30
    )
    
    # IR function signature
    function_sig = st.fixed_dictionaries({
        "name": identifier,
        "params": st.lists(
            st.fixed_dictionaries({
                "name": identifier,
                "type": ir_types
            }),
            max_size=5
        ),
        "return_type": ir_types,
        "body": st.lists(
            st.fixed_dictionaries({
                "op": st.sampled_from(["return", "add", "sub", "mul", "div", "load", "store"]),
                "args": st.lists(identifier, max_size=3)
            }),
            max_size=10
        )
    })
    
    # Full IR module
    return st.fixed_dictionaries({
        "module": identifier,
        "ir_version": st.just("1.0"),
        "functions": st.lists(function_sig, min_size=1, max_size=5),
        "types": st.lists(
            st.fixed_dictionaries({
                "name": identifier,
                "kind": st.sampled_from(["struct", "enum", "alias"]),
                "fields": st.lists(st.tuples(identifier, ir_types), max_size=5)
            }),
            max_size=3
        )
    })


def manifest_strategy():
    """Generate STUNIR manifest-like structures."""
    if not HYPOTHESIS_AVAILABLE:
        return None
    
    # SHA256 hash (64 hex chars)
    sha256_hash = st.text(
        alphabet="0123456789abcdef",
        min_size=64,
        max_size=64
    )
    
    # File path
    path = st.text(
        alphabet=string.ascii_letters + string.digits + "_-./",
        min_size=1,
        max_size=100
    )
    
    # Manifest entry
    entry = st.fixed_dictionaries({
        "name": st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=30),
        "path": path,
        "hash": sha256_hash,
        "size": st.integers(min_value=0, max_value=10**9),
    })
    
    # Full manifest
    return st.fixed_dictionaries({
        "manifest_schema": st.just("stunir.manifest.v1"),
        "manifest_epoch": st.integers(min_value=0, max_value=2**32),
        "manifest_hash": sha256_hash,
        "entries": st.lists(entry, min_size=0, max_size=20)
    })


def path_strategy():
    """Generate path-like strings for security testing."""
    if not HYPOTHESIS_AVAILABLE:
        return None
    
    # Mix of valid and potentially dangerous paths
    return st.one_of(
        # Normal paths
        st.text(alphabet=string.ascii_letters + string.digits + "_-./", max_size=200),
        # Path traversal attempts
        st.text().map(lambda s: f"../{s}"),
        st.text().map(lambda s: f"./{s}/../{s}"),
        # Null byte injection
        st.text(max_size=50).map(lambda s: f"{s}\x00.txt"),
        # Unicode edge cases
        st.text(alphabet="\u202e\u200b\u00a0" + string.ascii_letters, max_size=50),
        # Long paths
        st.text(alphabet=string.ascii_letters, min_size=200, max_size=500),
    )


def unicode_strategy():
    """Generate Unicode edge case strings."""
    if not HYPOTHESIS_AVAILABLE:
        return None
    
    return st.one_of(
        # Empty and whitespace
        st.just(""),
        st.text(alphabet=" \t\n\r", max_size=20),
        # Control characters
        st.text(alphabet="\x00\x01\x02\x03\x04\x05\x1b\x7f", max_size=20),
        # RTL and special Unicode
        st.text(alphabet="\u202e\u202d\u200b\u200c\u200d\ufeff", max_size=20),
        # Emoji and extended characters
        st.text(alphabet="\U0001f600\U0001f4a9\U0001f914", max_size=10),
        # Mixed normal text with edge cases
        st.text(max_size=100),
    )
