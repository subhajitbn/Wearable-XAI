"""
Suppress specific error messages and warnings, particularly from rpy2 and other libraries.
This module provides a context manager to suppress output and a function to handle rpy2-specific warnings.
"""
import warnings
import os
import sys
import logging
import contextlib

@contextlib.contextmanager
def suppress_output():
    """
    A context manager to suppress stdout and stderr.

    Redirects the standard output and error streams to os.devnull, effectively
    silencing any output during the execution of the code block within the 
    context. Restores the original streams upon exit.
    """

    with open(os.devnull, 'w', encoding='utf-8') as devnull:
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
    """
    Suppresses specific error messages and warnings from rpy2 and other libraries.

    This function suppresses warnings related to the rpy2 library, including the 
    PATH override warning and API import fallback warning. It temporarily redirects 
    stderr to silence specific error messages during the import of rpy2.robjects. 
    Additionally, it sets the logging level for the rpy2 logger to ERROR to reduce 
    noise in the logs.
    """
    
    # Suppress rpy2 PATH override warning
    warnings.filterwarnings("ignore", message='Environment variable "PATH" redefined by R.*')

    # pylint: disable=missing-class-docstring, missing-function-docstring
    # Suppress rpy2 API import fallback warning
    class NullStream:
        def write(self, *_): 
            pass
        def flush(self): 
            pass
    # pylint: enable=missing-class-docstring, missing-function-docstring
    
    # Temporarily suppress stderr to silence "Error importing in API mode"
    sys.stderr = NullStream()

    # pylint: disable=broad-exception-caught, consider-using-from-import, import-outside-toplevel, unused-import
    try:
        import rpy2.robjects as robjects  # Trigger R interface import
    except Exception:
        pass  # Optional: log it if needed
    # pylint: enable=broad-exception-caught, consider-using-from-import, import-outside-toplevel, unused-import
    
    # Restore stderr
    sys.stderr = sys.__stderr__

    # Optionally suppress logging noise
    logging.getLogger("rpy2").setLevel(logging.ERROR)
