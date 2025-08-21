DRRT_DIR=$(builtin pwd)
ENOKI_DIR="$DRRT_DIR/build/ext/enoki"
export PYTHONPATH="$ENOKI_DIR:$DRRT_DIR/build:$DRRT_DIR/build/lib:$PYTHONPATH"
export LD_LIBRARY_PATH="$ENOKI_DIR:$DRRT_DIR/build:$DRRT_DIR/build/lib:$LD_LIBRARY_PATH"
