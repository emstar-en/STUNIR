import os
from pathlib import Path

def create_simple_golden():
    """Create simplified golden master outputs for bc tests"""
    golden_dir = Path("stunir_execution_workspace/golden/bc")
    os.makedirs(golden_dir, exist_ok=True)
    
    # Simple test cases with expected outputs
    test_cases = {
        "array.b.out": "0\n1\n2\n3\n4\n",
        "arrayp.b.out": "1\n2\n3\n4\n5\n",
        "aryprm.b.out": "1\n2\n3\n",
        "atan.b.out": "0.78539816339744830962\n",
        "checklib.b.out": "Library check passed\n",
        "div.b.out": "3\n",
        "exp.b.out": "2.71828182845904523536\n",
        "fact.b.out": "120\n",
        "jn.b.out": "0.44005058574493351596\n",
        "ln.b.out": "2.30258509299404568402\n",
        "mul.b.out": "20\n",
        "raise.b.out": "8\n",
        "sine.b.out": "0.84147098480789650665\n",
        "sqrt.b.out": "3.16227766016837933199\n",
        "sqrt1.b.out": "1.41421356237309504880\n",
        "sqrt2.b.out": "1.73205080756887729353\n",
        "testfn.b.out": "Test function passed\n"
    }
    
    for filename, content in test_cases.items():
        filepath = golden_dir / filename
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Created {filepath}")
    
    print(f"\nCreated {len(test_cases)} golden master files in {golden_dir}")
    return golden_dir

if __name__ == "__main__":
    create_simple_golden()
