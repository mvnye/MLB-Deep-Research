"""A tiny runner: every test_ function runs, and failures are listed instead of stopping the file."""

import sys
import traceback


def run_tests(names: dict) -> None:
    tests = [f for name, f in names.items() if name.startswith("test_") and callable(f)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as e:  # an assert or a crash: both are a failed test
            failed += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc(limit=2)
    print(f"{len(tests) - failed} of {len(tests)} passed")
    sys.exit(1 if failed else 0)
