import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch_tiled"

# Same stack as the `torch` provider -- this is plain SDPA, called once per query
# tile instead of once per call -- so the version record is the same. It is a
# separate provider rather than a change to `ops/torch` because it wins only on
# the shapes that overflow HBM with their own score matrix and loses elsewhere;
# see flash_attention.py in this directory.
_provider_info = {}
for _pkg in ("torch-neuronx", "torch-xla"):
    try:
        _provider_info[_pkg] = importlib.metadata.version(_pkg)
    except importlib.metadata.PackageNotFoundError:
        pass

if _provider_info:
    ProviderRegistry.register_provider_info(PROVIDER_NAME, _provider_info)
