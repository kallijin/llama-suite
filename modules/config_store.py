import sys
from modules.llama_cpp import config_store as _impl

sys.modules[__name__] = _impl
