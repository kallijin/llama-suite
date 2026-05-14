import sys
from modules.llama_cpp import model_scan as _impl

sys.modules[__name__] = _impl
