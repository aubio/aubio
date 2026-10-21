# Copilot Setup Steps Documentation

This document explains the `.github/copilot-setup-steps.yml` file and how it provides a development environment setup for aubio-ledfx.

## Overview

The `copilot-setup-steps.yml` file is a GitHub Actions workflow that can be used to set up a complete development environment for aubio-ledfx. It installs all necessary tools and dependencies for building, testing, and developing the project.

## What Gets Installed

### 1. System Build Tools
- **Compilers**: gcc, g++ (C99-compliant for aubio)
- **Build systems**: ninja-build (used by Meson)
- **Build utilities**: pkg-config, git
- **Development libraries**:
  - libfftw3-dev (FFT library)
  - libsndfile1-dev (audio file I/O)
  - libsamplerate0-dev (sample rate conversion)
  - libjack-dev (JACK audio)
  - libavcodec-dev, libavformat-dev, libswresample-dev (FFmpeg)
  - librubberband-dev (pitch shifting/time stretching)

### 2. Python Build Environment
- **Python 3.11**: Set up via actions/setup-python
- **Meson**: >=1.9.0 (primary build system)
- **meson-python**: PEP 517 build backend for Python packages
- **ninja**: Build tool (also installed as Python package)
- **NumPy**: >=1.26.4 (required for aubio Python bindings)
- **pip, setuptools, wheel**: Updated to latest versions

### 3. Testing Tools
- **pytest**: Python test runner
- **pytest-cov**: Code coverage for tests
- **build**: Python package build tool
- **uv**: Modern, fast Python package manager

### 4. Security Testing Tools
- **libasan6**: AddressSanitizer runtime library
- **libubsan1**: UndefinedBehaviorSanitizer runtime library
- **valgrind**: Memory error detection tool

These tools enable running the security test suite defined in `.github/workflows/sanitizers.yml`.

### 5. Environment Configuration
Sets up environment variables:
- `PKG_CONFIG_PATH`: Includes system pkg-config paths

## Key Changes from Previous Version

**Removed vcpkg**: The setup now uses system-provided dependencies instead of building them from source with vcpkg. This simplifies the setup and makes it faster, as Ubuntu's package repository provides all necessary libraries.

**System Dependencies**: All C/C++ dependencies are now installed via apt-get:
- Audio codecs and formats via libavcodec-dev, libavformat-dev
- Rubberband for pitch/time manipulation via librubberband-dev
- No need to build these from source
### 6. Verification
- Displays versions of all installed tools
- Lists available system libraries via pkg-config
- Shows installed Python packages
- Provides usage instructions

## How It Works

The `copilot-setup-steps.yml` file is a standard GitHub Actions workflow that can be:

1. Run manually via workflow_dispatch
2. Referenced as a reusable workflow
3. Used as a template for setting up development environments

The workflow has a single job called "setup" that runs on ubuntu-latest and executes 8 steps to configure the environment.

## Usage Examples

After running the setup workflow (manually via GitHub Actions), you can:

### Build the C Library
```bash
meson setup builddir
meson compile -C builddir
```

### Build with Tests Enabled
```bash
meson setup builddir -Dtests=true
meson compile -C builddir
meson test -C builddir
```

### Build with Security Sanitizers
```bash
meson setup builddir -Db_sanitize=address,undefined -Dtests=true
meson compile -C builddir
meson test -C builddir
```

### Build Python Wheel
```bash
# Using uv (faster)
uv build --wheel

# Using pip
pip wheel . --no-deps
```

### Run Tests
```bash
# C tests
meson test -C builddir --print-errorlogs

# Python tests
pytest python/tests/
```

## Benefits

1. **Simplified Setup**: Uses system packages instead of building from source
2. **Faster Setup**: No need to compile dependencies with vcpkg
3. **Complete Toolchain**: All build, test, and security tools available
4. **Standard Workflow**: Follows GitHub Actions best practices

## Maintenance

### Adding New Tools

To add a new tool to the setup:

1. Edit `.github/copilot-setup-steps.yml`
2. Add a new step in the `jobs.setup.steps` section
3. Test the workflow by running it manually
4. Update this documentation

### Adding New Dependencies

For Python dependencies:
- Add to the pip install command in the "Install Python build tools" or "Install Python testing and development tools" step

For system packages:
- Add to the apt-get install command in the "Install system build tools and dependencies" step

## Platform Notes

This setup workflow is designed for **Ubuntu Linux** environments (ubuntu-latest in GitHub Actions).

For local development on other platforms:
- **macOS**: Use Homebrew for system packages (`brew install` instead of `apt-get`)
- **Windows**: Use vcpkg or system package managers, MSVC or MinGW-w64 for compilation

See the main README.md for platform-specific instructions.

## Related Files

- **`.github/copilot-instructions.md`**: Custom instructions for Copilot's behavior
- **`vcpkg.json`**: Declares C/C++ dependencies
- **`pyproject.toml`**: Python package configuration
- **`meson.build`**: Main build system configuration
- **`meson_options.txt`**: Build options and feature toggles
- **`.github/workflows/build.yml`**: CI/CD build workflow
- **`.github/workflows/sanitizers.yml`**: Security testing workflow

## Troubleshooting

### Workflow Fails to Run

If the workflow fails to run, check:
1. GitHub Actions permissions in repository settings
2. Workflow file syntax (validate YAML)
3. Internet connectivity (for downloading packages)

### Missing Dependencies

If dependencies are missing after setup:
```bash
# Manually install system packages
sudo apt-get update
sudo apt-get install -y libfftw3-dev libsndfile1-dev libsamplerate0-dev

# Verify packages are available
pkg-config --list-all | grep -E "fftw3|sndfile|samplerate"
```

## References

- [GitHub Actions Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Meson Build System](https://mesonbuild.com/)
- [meson-python](https://meson-python.readthedocs.io/)
- [aubio-ledfx Build Documentation](../doc/building.rst)
- [aubio-ledfx Meson Reference](../doc/meson_reference.rst)
