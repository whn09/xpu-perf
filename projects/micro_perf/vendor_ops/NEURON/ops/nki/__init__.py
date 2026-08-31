import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "nki"

try:
    ProviderRegistry.register_provider_info("nki", {
        "neuronxcc": importlib.metadata.version("neuronx-cc"),
    })
except Exception:
    pass
