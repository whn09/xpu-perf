import pathlib


def check_neuron_env():
    # NOTE: deliberately does not import torch_xla. Importing it here would
    # initialise the PJRT runtime in the parent process and claim every
    # NeuronCore, leaving none for the spawned worker processes. See the
    # module docstring of backend_neuron.py.
    devices = sorted(pathlib.Path("/dev").glob("neuron*"))
    if not devices:
        raise EnvironmentError(
            "No Neuron device found under /dev/neuron*. "
            "Check that the aws-neuronx driver is loaded (run `neuron-ls`)."
        )
    print(f"Neuron is available. Found {len(devices)} Neuron device(s).")


check_neuron_env()

from .backend_neuron import BackendNEURON
