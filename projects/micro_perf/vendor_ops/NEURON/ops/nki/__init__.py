import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "nki"

try:
    ProviderRegistry.register_provider_info(PROVIDER_NAME, {
        "neuronxcc": importlib.metadata.version("neuronx-cc"),
    })
except importlib.metadata.PackageNotFoundError:
    pass
