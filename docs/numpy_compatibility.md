# NumPy compatibility

The core dependency range accepts NumPy 1 and 2. Python 3.12 or later is
needed for NumPy 2.5; pip selects an older compatible version on older Python.
Use PyTorch 2.4 or later with NumPy 2: older PyTorch wheels may not provide
working NumPy array/tensor interoperability. The dependency markers preserve
the existing PyTorch range on Python versions that cannot install NumPy 2.
Setuptools 65.5–80 allows both the legacy Gym dependency and modern PyTorch.

The compatibility workflow installs dependencies normally, runs `pip check`,
and verifies array/tensor round trips and Gymnasium environment interaction.
For environments, create a Gymnasium instance and pass it to `DingEnvWrapper`.
The adapter preserves DI-engine's existing four-field timestep interface.

Legacy Gym 0.25.1 remains available for older environment integrations, but
its environment checker uses the removed `np.bool8` alias. Those legacy Gym
environments and optional third-party simulators are not covered by the NumPy
2 gate; use Gymnasium or retain a separate NumPy 1 environment for them.
No NumPy aliases are added or monkey-patched globally.
