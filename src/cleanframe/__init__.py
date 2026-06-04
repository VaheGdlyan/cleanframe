__version__ = "0.1.0"

from .accessor import CleanFrameAccessor  # This triggers the .cf registration on import
from .pipeline import DataCleaner

# Adding CleanFrameAccessor here satisfies the linter and exports it cleanly
__all__ = ["DataCleaner", "CleanFrameAccessor"]