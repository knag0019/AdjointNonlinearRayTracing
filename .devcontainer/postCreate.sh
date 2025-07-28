#!/usr/bin/env bash
set -euo pipefail

# ---- Git safe dir & submodules ----
git config --global --add safe.directory '*'
git submodule sync --recursive
git submodule update --init --recursive --jobs 8

# ---- Build ----
mkdir -p build
cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENOKI_CUDA=1 -DENOKI_AUTODIFF=1 -DENOKI_PYTHON=1 \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DPYTHON_EXECUTABLE:FILEPATH="$(which python)"
cmake --build . -j "$(nproc)"
cd ..

# ---- Path setup script ----
cat > setpath.sh <<'EOF'
DRRT_DIR=$(builtin pwd)
ENOKI_DIR="$DRRT_DIR/build/ext/enoki"
export PYTHONPATH="$ENOKI_DIR:$DRRT_DIR/build:$DRRT_DIR/build/lib:$PYTHONPATH"
export LD_LIBRARY_PATH="$ENOKI_DIR:$DRRT_DIR/build:$DRRT_DIR/build/lib:$LD_LIBRARY_PATH"
EOF

grep -q 'setpath.sh' ~/.bashrc || echo "source $(pwd)/setpath.sh" >> ~/.bashrc
