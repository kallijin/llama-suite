import sys
from modules.vllm import profile_store as _impl

sys.modules[__name__] = _impl
