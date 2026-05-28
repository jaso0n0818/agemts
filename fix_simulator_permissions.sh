#!/bin/sh
# Run once with sudo after vcpkg was bootstrapped as root:
#   sudo ./fix_simulator_permissions.sh
set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TRADING="$REPO_ROOT/simulate/trading"
TARGET_USER="${SUDO_USER:-$USER}"

chown -R "$TARGET_USER:$TARGET_USER" "$TRADING/vcpkg"
rm -rf "$TRADING/build" "$TRADING/build-user"
echo "OK: vcpkg owned by $TARGET_USER; old build dirs removed."
echo "Next:"
echo "  cd $TRADING && mkdir build && cd build"
echo "  cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_MAKE_PROGRAM=/usr/bin/make .."
echo "  cmake --build . -j\$(nproc)"
