import sys
from modules.llama_cpp import probes as _impl

sys.modules[__name__] = _impl
