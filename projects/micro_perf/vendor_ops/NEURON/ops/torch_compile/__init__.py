import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch_compile"

# This provider's numbers are a *compiler* result, so report the compiler as well
# as the framework. Look each up independently: neuronx-cc ships without dist-info
# on some images, and one absence should not discard the versions we do know.
_provider_info = {}
for _pkg in ("torch", "torch-neuronx", "neuronx-cc", "nki"):
    try:
        _provider_info[_pkg] = importlib.metadata.version(_pkg)
    except importlib.metadata.PackageNotFoundError:
        pass

if _provider_info:
    ProviderRegistry.register_provider_info(PROVIDER_NAME, _provider_info)
