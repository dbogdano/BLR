#!/usr/bin/env python3
"""
Quick syntax check for the modified tagfastq.py
This script verifies that the changes don't break basic import/parse.
"""
import sys
import ast

def check_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Try to parse the file as an AST
        ast.parse(code)
        print(f"✓ {filepath} has valid Python syntax")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error in {filepath}:")
        print(f"  Line {e.lineno}: {e.msg}")
        print(f"  {e.text}")
        return False
    except Exception as e:
        print(f"✗ Error checking {filepath}: {e}")
        return False

if __name__ == "__main__":
    filepath = "src/blr/cli/tagfastq.py"
    success = check_syntax(filepath)
    sys.exit(0 if success else 1)
