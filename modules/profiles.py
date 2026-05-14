import sys
from modules.llama_cpp import profiles as _impl

sys.modules[__name__] = _impl
