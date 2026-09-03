import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "nki"

# Two NKI namespaces can coexist in one image: the standalone `nki` package (what
# nkilib and the native runtime use) and `neuronxcc.nki` (bundled with the compiler,
# HLO-traced, XLA only). This provider's kernel comes from nkilib, which is built on
# the standalone one, so report both rather than just the compiler version.
_info = {}
for _dist, _key in (("nki", "nki"), ("neuronx-cc", "neuronxcc")):
    try:
        _info[_key] = importlib.metadata.version(_dist)
    except importlib.metadata.PackageNotFoundError:
        pass

if _info:
    ProviderRegistry.register_provider_info(PROVIDER_NAME, _info)
