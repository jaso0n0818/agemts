#!/bin/sh
# Build taosim as normal user (after fix_simulator_permissions.sh).
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRADING="$REPO_ROOT/simulate/trading"

cd "$TRADING"
rm -rf build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_MAKE_PROGRAM=/usr/bin/make \
  ..
cmake --build . -j"$(nproc)"
ls -la src/cpp/taosim
echo "SUCCESS: $TRADING/build/src/cpp/taosim"
