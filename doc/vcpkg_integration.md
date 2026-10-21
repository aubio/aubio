# vcpkg Integration for aubio-ledfx

This document explains the vcpkg integration for cross-platform dependency management.

## Overview

aubio-ledfx uses [vcpkg](https://vcpkg.io) to manage external dependencies across all platforms (Linux, macOS, Windows). This replaces the previous platform-specific package managers (yum, brew, vcpkg manual commands) with a unified manifest-based approach.

## Files

### vcpkg.json
The vcpkg manifest file that declares all dependencies:
- **Core dependencies**: pkgconf, libsndfile, libsamplerate, fftw3
- These are installed automatically when building aubio

### meson-vcpkg.txt
A Meson native file that configures vcpkg integration (optional, mainly for documentation).

### meson.build
The main build file now includes vcpkg integration:
- Automatically detects vcpkg executable
- Determines the appropriate triplet (x64-windows, arm64-osx, etc.)
- Runs `vcpkg install` to fetch dependencies
- Configures pkg-config paths for dependency discovery

## How It Works

### CI/CD (GitHub Actions)

1. **vcpkg Installation**: Each runner installs vcpkg via git clone and bootstrap
2. **Manifest Mode**: vcpkg reads `vcpkg.json` and installs declared dependencies
3. **Triplet Detection**: Meson automatically selects the correct vcpkg triplet based on OS and architecture
4. **Dependency Discovery**: Meson finds dependencies via pkg-config (provided by vcpkg)
5. **Wheel Building**: cibuildwheel builds wheels with bundled dependencies

### Local Development

```bash
# Install vcpkg (if not already installed)
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh  # or .bat on Windows

# Set environment variable
export VCPKG_ROOT=/path/to/vcpkg

# Build aubio (vcpkg will install dependencies automatically)
meson setup builddir
meson compile -C builddir
```

## Benefits

1. **Unified Dependencies**: Same dependencies on all platforms
2. **Reproducible Builds**: vcpkg.json locks dependency versions
3. **No Manual Installation**: Dependencies install automatically
4. **Cross-Platform**: Works on Linux, macOS, Windows without changes
5. **CI/CD Simplification**: Removed platform-specific before-all commands

## Supported Triplets

- **Linux**: `x64-linux-pic`, `arm64-linux-pic` (custom triplets with -fPIC for static linking)
- **macOS**: `x64-osx`, `arm64-osx`
- **Windows**: `x64-windows-release` (release-only builds)

## Platform-Specific Dependencies

vcpkg.json uses platform expressions to conditionally install dependencies:

**All platforms:**
- pkgconf, libsndfile[external-libs], libsamplerate, fftw3

**Linux only:**
- mpg123, mp3lame, opus, libogg, libvorbis, libFLAC
- These are transitive dependencies of libsndfile that must be explicitly installed for static linking
- rubberband and ffmpeg are excluded due to PIC issues with assembly code

**Windows & macOS only:**
- rubberband (pitch shifting, time stretching)
- ffmpeg (avcodec, avformat, swresample)

Example platform expression in vcpkg.json:
```json
{
  "name": "rubberband",
  "platform": "!linux"
}
```

## Adding Dependencies

To add a new dependency:

1. Edit `vcpkg.json` to add the package name
2. Use platform expressions if needed: `"platform": "linux"` or `"platform": "!linux"`
3. Check vcpkg port exists: `vcpkg search <package>`
4. Update `meson.build` to use the dependency
5. Test locally and in CI

## Troubleshooting

### vcpkg not found
Ensure vcpkg is in PATH or set VCPKG_ROOT environment variable.

### Dependencies not installing
Check `vcpkg install` output in meson-log.txt for errors.

### Wrong triplet
Verify host_machine.system() and host_machine.cpu_family() detection in meson.build.

## Migration from Previous System

**Before**: Platform-specific package managers in pyproject.toml
```toml
[tool.cibuildwheel.linux]
before-all = "yum install -y libsndfile-devel ..."

[tool.cibuildwheel.macos]
before-all = "brew install libsndfile ..."

[tool.cibuildwheel.windows]
before-all = "vcpkg install libsndfile:x64-windows ..."
```

**After**: Unified vcpkg manifest
```json
{
  "dependencies": [
    "libsndfile",
    "libsamplerate",
    "fftw3"
  ]
}
```

## Performance

- **First build**: Downloads and builds dependencies (~5-10 minutes)
- **Subsequent builds**: Uses cached vcpkg packages (~30 seconds)
- **CI caching**: GitHub Actions caches vcpkg_installed directory

## References

- [vcpkg Documentation](https://vcpkg.io/en/docs/README.html)
- [vcpkg Manifest Mode](https://vcpkg.io/en/docs/users/manifests.html)
- [meson-vcpkg Example](https://github.com/Neumann-A/meson-vcpkg)
