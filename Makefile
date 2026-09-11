BUILD_DIR ?= build
CMAKE ?= cmake
CMAKE_BUILD_TYPE ?= Release
JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Portable binaries for distribution (no -march=native)
PORTABLE ?= 0
ifeq ($(PORTABLE),1)
  CMAKE_EXTRA_FLAGS += -DKAHU_NATIVE_ARCH=OFF
endif

.PHONY: configure kuba_engine clean distclean \
        release cross-%

configure:
	$(CMAKE) -B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(CMAKE_BUILD_TYPE) \
		$(CMAKE_EXTRA_FLAGS)

kuba_engine: configure
	$(CMAKE) --build $(BUILD_DIR) --target kuba_engine -j$(JOBS)

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean 2>/dev/null || true

distclean:
	rm -rf $(BUILD_DIR) gui/build kuba_engine nnue_trainer

# Portable release build for the current machine (suitable for GitHub upload)
release:
	$(MAKE) all BUILD_DIR=$(BUILD_DIR) CMAKE_BUILD_TYPE=Release PORTABLE=1

# Cross-target release builds: make cross-linux-x86_64, cross-macos-arm64, etc.
cross-%: PORTABLE = 1
cross-%:
	@./scripts/cross-build.sh $*
