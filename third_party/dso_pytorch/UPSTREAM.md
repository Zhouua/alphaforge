# Vendored PyTorch DSO

Source: https://github.com/dso-org/deep-symbolic-optimization-pytorch

Vendored commit: `85f88faf3ff95636f0011764b30e13f7bb82b54c`

The upstream repository is BSD-3-Clause licensed; its `LICENSE` and `NOTICE`
files are retained here.

Do not run `pip install -e third_party/dso_pytorch/dso` in the shared
RD-Agent environment. Upstream `setup.py` still pins legacy NumPy and Numba
versions that do not support Python 3.11. This project imports the source tree
directly and installs only the compatible runtime packages listed in
`requirements-dso-pytorch.txt`.

Local compatibility changes:

- use NumPy 2-compatible `np.bool_` and `np.int64` aliases;
- use NumPy 2-compatible array byte keys and quantile APIs;
- allow the search routines to fall back to Python when Numba is unavailable;
- fall back to the pure-Python executor when the optional Cython extension is
  not built.

Upstream images, tests, control-policy checkpoints, and the optional pretrained
language-model checkpoint are omitted because this project only uses the
custom Qlib symbolic task.
