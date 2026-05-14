import sys
from modules.vllm import api_probe as _impl

sys.modules[__name__] = _impl
