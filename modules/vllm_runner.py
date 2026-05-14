import sys
from modules.vllm import runner as _impl

sys.modules[__name__] = _impl
