"""MapleHelper: one-button escape combos with an on-screen coach for a dart thief."""

import os

# OpenBLAS (numpy's BLAS, environment.yml) commits a work buffer per thread when numpy loads: about 3 GB a Python
# process with this PC's 24 threads (18 MB with one), and a few processes ran it out of commit charge on 2026-10-24.
# Nothing here needs threaded BLAS; set before numpy is first imported (the env's sitecustomize.py does it too).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
