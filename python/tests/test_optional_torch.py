import subprocess
import sys


def test_base_import_and_missing_torch_error():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch intentionally unavailable", name="torch")

sys.meta_path.insert(0, BlockTorch())
import lazymatrix
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
try:
    import lazymatrix.torch
except ModuleNotFoundError as error:
    assert error.name == "torch"
    assert 'pip install "lazymatrix[torch]"' in str(error)
else:
    raise AssertionError("adapter import should explain the missing extra")
""",
        ],
        check=True,
    )
