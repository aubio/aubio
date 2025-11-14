# Custom triplet for x64 Linux with position-independent code
# Required to link static FFmpeg libraries into shared Python extension

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Force position-independent code for all static libraries
# This is critical for linking into shared objects (.so files)
set(VCPKG_C_FLAGS "-fPIC")
set(VCPKG_CXX_FLAGS "-fPIC")
