import sys
from modules.vllm import doctor as _impl

sys.modules[__name__] = _impl
