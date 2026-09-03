import importlib.metadata
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "nkilib"

# Kernels called directly out of the `nkilib` package that ships in the
# PyTorch-native Neuron images -- as opposed to `ops/nki`, whose kernel comes from
# `neuronxcc.nki.kernels` and only runs under torch_xla. nkilib is built on the
# standalone `nki` package, and the kernel is compiled by neuronx-cc, so all three
# belong in the version record. `nkilib` itself has shipped without dist-info on
# some images, hence the independent lookups.
_provider_info = {}
for _pkg in ("nkilib", "nki", "neuronx-cc", "torch-neuronx"):
    try:
        _provider_info[_pkg] = importlib.metadata.version(_pkg)
    except importlib.metadata.PackageNotFoundError:
        pass

if _provider_info:
    ProviderRegistry.register_provider_info(PROVIDER_NAME, _provider_info)
