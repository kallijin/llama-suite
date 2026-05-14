import sys
from modules.llama_cpp import backend_inspector as _impl

sys.modules[__name__] = _impl
