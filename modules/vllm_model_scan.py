import sys
from modules.vllm import model_scan as _impl

sys.modules[__name__] = _impl
