# Custom triplet for ARM64 Linux with PIC enabled for static libraries
# This is required for linking static libraries into shared libraries (Python extensions)

set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Force position-independent code for all static libraries
# This is critical for FFmpeg and other dependencies when linking into shared libraries
set(VCPKG_C_FLAGS "-fPIC")
set(VCPKG_CXX_FLAGS "-fPIC")
