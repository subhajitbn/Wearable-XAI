import warnings
import os
import sys
import logging
import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def suppress_rpy2_and_other_errmsg():
    # Suppress rpy2 PATH override warning
    warnings.filterwarnings("ignore", message='Environment variable "PATH" redefined by R.*')

    # Suppress rpy2 API import fallback warning
    class NullStream:
        def write(self, *_): pass
        def flush(self): pass

    # Temporarily suppress stderr to silence "Error importing in API mode"
    sys.stderr = NullStream()

    try:
        import rpy2.robjects as robjects  # Trigger R interface import
    except Exception:
        pass  # Optional: log it if needed

    # Restore stderr
    sys.stderr = sys.__stderr__

    # Optionally suppress logging noise
    logging.getLogger("rpy2").setLevel(logging.ERROR)
