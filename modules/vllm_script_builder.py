import sys
from modules.vllm import script_builder as _impl

sys.modules[__name__] = _impl
