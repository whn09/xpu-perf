import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch"

# Look each package up independently. torch-neuronx is not installed on every
# Neuron stack -- the vLLM inference venv drives torch_xla and neuronx-cc
# directly -- and building one dict would let that single absence discard the
# versions we do know.
_provider_info = {}
for _pkg in ("torch-neuronx", "torch-xla"):
    try:
        _provider_info[_pkg] = importlib.metadata.version(_pkg)
    except importlib.metadata.PackageNotFoundError:
        pass

if _provider_info:
    ProviderRegistry.register_provider_info(PROVIDER_NAME, _provider_info)
