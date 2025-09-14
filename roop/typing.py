from typing import Any
import numpy

# Try to import insightface, fall back to mock if not available
try:
    from insightface.app.common import Face
except ImportError:
    print("insightface not found, using mock implementation")
    from roop.mock_insightface import Face

Frame = numpy.ndarray[Any, Any]
