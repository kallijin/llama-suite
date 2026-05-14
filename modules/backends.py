import sys
from modules.llama_cpp import backends as _impl

sys.modules[__name__] = _impl
