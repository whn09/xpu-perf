import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch"

try:
    ProviderRegistry.register_provider_info("torch", {
        "torch-neuronx": importlib.metadata.version("torch-neuronx"),
        "torch-xla": importlib.metadata.version("torch-xla"),
    })
except Exception:
    pass
