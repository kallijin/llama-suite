import sys
from modules.vllm import profiles as _impl

sys.modules[__name__] = _impl
