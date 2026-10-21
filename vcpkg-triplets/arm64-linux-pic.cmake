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

# Release only: wheels never link the debug libraries
set(VCPKG_BUILD_TYPE release)

# mpg123 asks CMake whether the CPU has an FPU, which on Linux reads
# /proc/cpuinfo; aarch64 lists "fp", not "fpu", so the answer is no and
# mpg123 refuses to build ("Bad decoder choice together with fixed point
# math!"). Every aarch64 CPU has one; vcpkg's port takes HAVE_FPU from here.
if(PORT STREQUAL "mpg123")
    list(APPEND VCPKG_CMAKE_CONFIGURE_OPTIONS -DHAVE_FPU=1)
endif()
