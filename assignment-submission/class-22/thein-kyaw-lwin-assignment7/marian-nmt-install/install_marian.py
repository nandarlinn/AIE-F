"""
install_marian.py
=================
Kaggle kernel script: compiles Marian-NMT from source on Ubuntu 22.04 + CUDA
and copies the resulting binaries to /kaggle/working/marian-bins/ so they can
be saved as a Kaggle Dataset output for reuse in future training kernels.

Reference: https://marian-nmt.github.io/docs/
Environment: Ubuntu 22.04, 2x Tesla T4, CUDA 13.0, 4-core Intel Xeon
"""

import subprocess
import sys
import os
import shutil

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, **kwargs):
    """Run a shell command, stream output, and raise on failure."""
    print(f"\n>>> {cmd}\n{'='*70}")
    result = subprocess.run(
        cmd, shell=True, check=True,
        stdout=sys.stdout, stderr=sys.stderr,
        **kwargs
    )
    return result


def section(title):
    print(f"\n{'#'*70}")
    print(f"# {title}")
    print(f"{'#'*70}\n")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKING_DIR   = "/kaggle/working"
MARIAN_SRC    = os.path.join(WORKING_DIR, "marian")
MARIAN_BUILD  = os.path.join(MARIAN_SRC, "build")
MARIAN_BINS   = os.path.join(WORKING_DIR, "marian-bins")   # Kaggle dataset output


# ---------------------------------------------------------------------------
# Step 1: System dependencies
# ---------------------------------------------------------------------------

section("Step 1: Install system dependencies")

run("apt-get update -qq")
run(
    "apt-get install -y -qq "
    "git cmake build-essential "
    "libgoogle-perftools-dev "   # TCMalloc (speeds up CPU-side memory alloc)
    "ccache"                     # optional: speeds up any re-builds
)


# ---------------------------------------------------------------------------
# Step 2: Clone Marian
# ---------------------------------------------------------------------------

section("Step 2: Clone Marian-NMT from GitHub")

if os.path.exists(MARIAN_SRC):
    print(f"[info] {MARIAN_SRC} already exists, skipping clone.")
else:
    run(f"git clone --depth 1 https://github.com/marian-nmt/marian {MARIAN_SRC}")

run(f"git -C {MARIAN_SRC} log --oneline -5")


# ---------------------------------------------------------------------------
# Step 3: CMake configure
# ---------------------------------------------------------------------------

section("Step 3: CMake configure (GPU build, no SentencePiece, no web server)")

os.makedirs(MARIAN_BUILD, exist_ok=True)

cmake_flags = " ".join([
    "-DCMAKE_BUILD_TYPE=Release",
    "-DUSE_SENTENCEPIECE=off",    # using marian-vocab with word-level vocab
    "-DCOMPILE_SERVER=off",       # no web server needed
    "-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda",
])

run(f"cmake {cmake_flags} ..", cwd=MARIAN_BUILD)


# ---------------------------------------------------------------------------
# Step 4: Compile  (4 CPU cores available on Kaggle)
# ---------------------------------------------------------------------------

section("Step 4: Compile Marian (make -j4) — expect ~10-15 minutes")

run("make -j4", cwd=MARIAN_BUILD)


# ---------------------------------------------------------------------------
# Step 5: Verify binaries
# ---------------------------------------------------------------------------

section("Step 5: Verify Marian binaries")

BINARIES = [
    "marian",
    "marian-decoder",
    "marian-vocab",
    "marian-scorer",
    "marian-conv",
]

for binary in BINARIES:
    binary_path = os.path.join(MARIAN_BUILD, binary)
    if os.path.isfile(binary_path):
        print(f"[OK] Found: {binary_path}")
        # Run --help (exits 0 for marian; exit 1 is fine too — just want it to execute)
        try:
            result = subprocess.run(
                [binary_path, "--help"],
                capture_output=True, text=True
            )
            # Print first 3 lines of help output as confirmation
            first_lines = result.stdout.strip().split("\n")[:3]
            for line in first_lines:
                print(f"    {line}")
        except Exception as e:
            print(f"[WARN] Could not run {binary}: {e}")
    else:
        print(f"[MISSING] {binary_path} — build may have failed!")


# ---------------------------------------------------------------------------
# Step 6: Copy binaries to Kaggle output directory
# ---------------------------------------------------------------------------

section("Step 6: Save binaries to /kaggle/working/marian-bins/")

os.makedirs(MARIAN_BINS, exist_ok=True)

for binary in BINARIES:
    src_path = os.path.join(MARIAN_BUILD, binary)
    dst_path = os.path.join(MARIAN_BINS, binary)
    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
        os.chmod(dst_path, 0o755)
        size_mb = os.path.getsize(dst_path) / (1024 * 1024)
        print(f"[saved] {dst_path}  ({size_mb:.1f} MB)")
    else:
        print(f"[skip]  {binary} not found, skipping copy.")

print(f"\n[done] All available binaries saved to {MARIAN_BINS}/")
print("[info] Save /kaggle/working/marian-bins/ as a Kaggle Dataset")
print("[info] Future training kernels can add that dataset and use:")
print(f"       /kaggle/input/<dataset-name>/marian")
print(f"       /kaggle/input/<dataset-name>/marian-decoder")
print(f"       /kaggle/input/<dataset-name>/marian-vocab")


# ---------------------------------------------------------------------------
# Step 7: Final summary
# ---------------------------------------------------------------------------

section("Step 7: Build summary")

run(f"ls -lh {MARIAN_BINS}/")
run(f"{MARIAN_BINS}/marian --version", cwd=WORKING_DIR)

print("\n[SUCCESS] Marian-NMT installation complete!")
print("Next: save the kernel output as a Kaggle Dataset named e.g. 'marian-nmt'")
