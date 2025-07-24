#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner

This script runs the comprehensive test suite for the recovery functionality,
including database generation, integration tests, performance tests, and
boundary condition tests.
"""

import subprocess
import sys
import time
from typing import List, Tuple


def run_test_suite(test_pattern: str, description: str) -> Tuple[bool, float]:
    """
    Run a test suite and return success status and duration.
    
    Args:
        test_pattern: pytest pattern to run
        description: Description of the test suite
        
    Returns:
        Tuple[bool, float]: (success, duration_seconds)
    """
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_pattern, "-v"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} PASSED ({duration:.2f}s)")
            print(f"Tests run: {result.stdout.count('PASSED')}")
            return True, duration
        else:
            print(f"❌ {description} FAILED ({duration:.2f}s)")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False, duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {description} TIMED OUT ({duration:.2f}s)")
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {description} ERROR: {str(e)} ({duration:.2f}s)")
        return False, duration


def main():
    """Run the comprehensive test suite."""
    print("🚀 Starting Comprehensive Recovery Test Suite")
    print(f"Python: {sys.executable}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define test suites to run
    test_suites = [
        ("tests/test_database_generator.py", "Database Generation Utilities"),
        ("tests/test_comprehensive_integration.py::TestComprehensiveIntegration", "End-to-End Integration Tests"),
        ("tests/test_comprehensive_integration.py::TestBoundaryConditions", "Boundary Condition Tests"),
        ("tests/test_performance.py::TestPerformance::test_recovery_performance_small_database", "Basic Performance Test"),
        ("tests/test_performance.py::TestPerformance::test_worker_scaling_performance", "Worker Scaling Test"),
        ("tests/test_performance.py::TestPerformanceRegression::test_baseline_recovery_performance", "Performance Regression Test"),
    ]
    
    # Run test suites
    results = []
    total_start_time = time.time()
    
    for test_pattern, description in test_suites:
        success, duration = run_test_suite(test_pattern, description)
        results.append((description, success, duration))
    
    total_duration = time.time() - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    print(f"Total Test Suites: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Total Duration: {total_duration:.2f}s")
    print()
    
    # Detailed results
    for description, success, duration in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {description:<40} ({duration:.2f}s)")
    
    # Final status
    if passed_count == total_count:
        print(f"\n🎉 ALL TEST SUITES PASSED!")
        return 0
    else:
        print(f"\n💥 {total_count - passed_count} TEST SUITE(S) FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())