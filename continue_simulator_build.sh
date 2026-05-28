#!/bin/sh
# Resume simulator build after install_simulator.sh stopped (e.g. at "python: not found").
# Prerequisites: vcpkg bootstrapped under simulate/trading/vcpkg
# Run: sudo ./continue_simulator_build.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT/simulate/trading"

if [ ! -x "./vcpkg/vcpkg" ]; then
	echo "ERROR: vcpkg not ready. Run: sudo ./install_simulator.sh (or bootstrap vcpkg first)"
	exit 1
fi

. /etc/lsb-release
echo "Ubuntu $DISTRIB_RELEASE — building taosim"

install_gpp14() {
	if g++ -dumpversion 2>/dev/null | grep -q "14"; then
		return 0
	fi
	if command -v g++-14 >/dev/null 2>&1; then
		echo "g++-14 already installed"
		return 0
	fi
	echo "Trying apt g++-14 (faster than source build)..."
	apt-get install -y software-properties-common || true
	add-apt-repository -y ppa:ubuntu-toolchain-r/test 2>/dev/null || true
	apt-get update
	if apt-get install -y g++-14 2>/dev/null; then
		return 0
	fi
	echo "apt g++-14 unavailable — compiling GCC 14.1 from source (1–3+ hours)..."
	apt-get install -y libmpfr-dev libgmp3-dev libmpc-dev wget
	wget -q http://ftp.gnu.org/gnu/gcc/gcc-14.1.0/gcc-14.1.0.tar.gz
	tar -xf gcc-14.1.0.tar.gz
	cd gcc-14.1.0
	./configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu \
		--prefix=/usr/local/gcc-14.1.0 --enable-checking=release --enable-languages=c,c++ \
		--disable-multilib --program-suffix=-14.1.0
	make -j"$(nproc)"
	make install
	cd ..
	rm -rf gcc-14.1.0 gcc-14.1.0.tar.gz
	update-alternatives --install /usr/bin/g++-14 g++-14 /usr/local/gcc-14.1.0/bin/g++-14.1.0 14 || true
}

install_cmake329() {
	if cmake --version 2>/dev/null | grep -q "3.29.7"; then
		return 0
	fi
	echo "Installing cmake 3.29.7 (may take 20–40 min)..."
	apt-get purge -y cmake || true
	wget -q https://github.com/Kitware/CMake/releases/download/v3.29.7/cmake-3.29.7.tar.gz
	tar zxf cmake-3.29.7.tar.gz
	cd cmake-3.29.7
	./bootstrap
	make -j"$(nproc)"
	make install
	cd ..
	rm -f cmake-3.29.7.tar.gz
	rm -rf cmake-3.29.7
}

install_gpp14
install_cmake329

rm -rf build
mkdir build
cd build
if g++ -dumpversion 2>/dev/null | grep -q "14"; then
	cmake -DCMAKE_BUILD_TYPE=Release ..
else
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14 ..
fi
cmake --build . -j"$(nproc)"

echo ""
echo "SUCCESS: $(pwd)/src/cpp/taosim"
ls -la src/cpp/taosim
