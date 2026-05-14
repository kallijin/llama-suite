import sys
from modules.llama_cpp import script_builder as _impl

sys.modules[__name__] = _impl
