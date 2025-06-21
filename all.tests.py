#!/usr/bin/env python3
"""
All Tests Runner for Grey Lord

This script discovers and runs all *.tests.py files in the repository.
It provides a unified way to run the complete test suite.

Usage:
    python all.tests.py
    python all.tests.py -v  (verbose output)
    python all.tests.py --pattern "*test*"  (custom pattern)
"""

import unittest
import sys
import os
from pathlib import Path
import argparse


def discover_and_run_tests(pattern="*.tests.py", start_dir=".", verbose=False):
    """
    Discover and run all test files matching the given pattern.
    
    Args:
        pattern: File pattern to match (default: "*.tests.py")
        start_dir: Directory to start discovery from (default: ".")
        verbose: Whether to run tests in verbose mode
        
    Returns:
        TestResult object with test results
    """
    # Set up the test loader
    loader = unittest.TestLoader()
    
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Find all test files
    test_files = []
    current_file = os.path.abspath(__file__)
    
    for root, dirs, files in os.walk(start_dir):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != '__pycache__']
        
        for file in files:
            if file.endswith('.tests.py'):
                test_file_path = os.path.abspath(os.path.join(root, file))
                # Skip the current file (all.tests.py)
                if test_file_path != current_file:
                    test_files.append(test_file_path)
    
    if not test_files:
        print(f"❌ No test files found matching pattern '{pattern}' in '{start_dir}'")
        return None
    
    print(f"🔍 Found {len(test_files)} test files:")
    for test_file in sorted(test_files):
        rel_path = os.path.relpath(test_file, start_dir)
        print(f"   📄 {rel_path}")
    print()
    
    # Import and add each test module
    successful_imports = 0
    failed_imports = []
    
    for test_file in test_files:
        try:
            # Convert file path to module path
            rel_path = os.path.relpath(test_file, start_dir)
            module_path = rel_path.replace(os.sep, '.').replace('.py', '')
            
            # Add the directory to sys.path if needed
            test_dir = os.path.dirname(os.path.abspath(test_file))
            if test_dir not in sys.path:
                sys.path.insert(0, test_dir)
            
            # Import the module dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location(module_path, test_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Discover tests in the module
            module_suite = loader.loadTestsFromModule(module)
            suite.addTest(module_suite)
            
            successful_imports += 1
            print(f"✅ Loaded tests from: {rel_path}")
            
        except Exception as e:
            failed_imports.append((test_file, str(e)))
            print(f"❌ Failed to load: {os.path.relpath(test_file, start_dir)}")
            print(f"   Error: {e}")
    
    print(f"\n📊 Test Discovery Summary:")
    print(f"   ✅ Successfully loaded: {successful_imports}")
    print(f"   ❌ Failed to load: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\n⚠️  Failed imports:")
        for file_path, error in failed_imports:
            rel_path = os.path.relpath(file_path, start_dir)
            print(f"   • {rel_path}: {error}")
    
    if successful_imports == 0:
        print(f"\n❌ No tests to run!")
        return None
    
    print(f"\n🚀 Running Tests...")
    print("=" * 80)
    
    # Run the tests
    runner = unittest.TextTestRunner(
        verbosity=2 if verbose else 1,
        stream=sys.stdout,
        buffer=True
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("=" * 80)
    print(f"📊 Test Results Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   ✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Failed: {len(result.failures)}")
    print(f"   💥 Errors: {len(result.errors)}")
    print(f"   ⏭️  Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split(chr(10))[0]}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.split(chr(10))[0]}")
    
    # Return appropriate exit code
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'🎉 All tests passed!' if success else '💀 Some tests failed!'}")
    
    return result


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Grey Lord Test Runner - Run all tests in the repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python all.tests.py                    # Run all *.tests.py files
    python all.tests.py -v                 # Run with verbose output
    python all.tests.py --pattern "*test*" # Custom file pattern
    python all.tests.py --dir training     # Run tests in specific directory
        """
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Run tests in verbose mode"
    )
    
    parser.add_argument(
        "--pattern",
        default="*.tests.py",
        help="File pattern to match (default: *.tests.py)"
    )
    
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory to start test discovery from (default: current directory)"
    )
    
    args = parser.parse_args()
    
    print("🧪 Grey Lord Test Runner")
    print("=" * 40)
    print(f"Pattern: {args.pattern}")
    print(f"Directory: {os.path.abspath(args.dir)}")
    print(f"Verbose: {args.verbose}")
    print()
    
    # Run the tests
    result = discover_and_run_tests(
        pattern=args.pattern,
        start_dir=args.dir,
        verbose=args.verbose
    )
    
    if result is None:
        return 1
    
    # Exit with appropriate code
    success = len(result.failures) == 0 and len(result.errors) == 0
    return 0 if success else 1


if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    sys.exit(main()) 