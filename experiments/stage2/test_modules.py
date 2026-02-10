"""
Test module for importing and testing Stage 2 implementations.

This script tests that different implementations can be imported correctly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

def test_cursor_imports():
    """Test Cursor implementation imports"""
    print("Testing Cursor implementation imports...")
    try:
        from experiments.stage2.cursor.register_tokens import RegisterTokenModule
        from experiments.stage2.cursor.loss import RegisterTokenLoss
        print("✓ Cursor imports successful")
        print(f"  - RegisterTokenModule: {RegisterTokenModule}")
        print(f"  - RegisterTokenLoss: {RegisterTokenLoss}")
        return True
    except Exception as e:
        print(f"✗ Cursor imports failed: {e}")
        return False

def test_cursor_package_import():
    """Test importing from cursor package"""
    print("\nTesting Cursor package import...")
    try:
        from experiments.stage2.cursor import RegisterTokenModule, RegisterTokenLoss
        print("✓ Cursor package import successful")
        return True
    except Exception as e:
        print(f"✗ Cursor package import failed: {e}")
        return False

def main():
    print("=" * 80)
    print("Stage 2 Implementation Import Tests")
    print("=" * 80)
    
    results = []
    
    # Test Cursor implementation
    print("\n[CURSOR IMPLEMENTATION]")
    print("-" * 80)
    results.append(("Cursor", test_cursor_imports()))
    results.append(("Cursor Package", test_cursor_package_import()))
    
    # Add tests for other implementations here as they are added
    # Example:
    # print("\n[OTHER IMPLEMENTATION]")
    # print("-" * 80)
    # results.append(("Other", test_other_imports()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✓ All imports successful!")
    else:
        print("\n✗ Some imports failed. Check errors above.")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
