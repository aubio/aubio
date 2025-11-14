# Memory Safety Testing with Sanitizers

This document describes how to use sanitizers to detect memory safety issues in aubio-ledfx.

## Overview

Sanitizers are runtime analysis tools that detect various types of bugs:

- **AddressSanitizer (ASAN)** - Detects memory errors (buffer overflows, use-after-free, etc.)
- **UndefinedBehaviorSanitizer (UBSAN)** - Detects undefined behavior
- **MemorySanitizer (MSAN)** - Detects uninitialized memory reads
- **ThreadSanitizer (TSAN)** - Detects data races in multithreaded code
- **LeakSanitizer (LSAN)** - Detects memory leaks (included with ASAN)

## Quick Start

### AddressSanitizer (Recommended for Most Use Cases)

```bash
# Setup build with AddressSanitizer
meson setup builddir-asan -Db_sanitize=address -Dtests=true

# Compile
meson compile -C builddir-asan

# Run tests
meson test -C builddir-asan --print-errorlogs
```

### UndefinedBehaviorSanitizer

```bash
# Setup build with UBSan
meson setup builddir-ubsan -Db_sanitize=undefined -Dtests=true

# Compile and test
meson compile -C builddir-ubsan
meson test -C builddir-ubsan --print-errorlogs
```

### Combined ASAN + UBSAN (Most Comprehensive)

```bash
# Setup build with both sanitizers
meson setup builddir-sanitizers -Db_sanitize=address,undefined -Dtests=true

# Compile and test
meson compile -C builddir-sanitizers
meson test -C builddir-sanitizers --print-errorlogs
```

## Detailed Usage

### AddressSanitizer (ASAN)

**Detects:**
- Heap buffer overflow
- Stack buffer overflow
- Global buffer overflow
- Use-after-free
- Use-after-return
- Use-after-scope
- Double-free
- Memory leaks (via LeakSanitizer)

**Setup:**
```bash
meson setup builddir-asan \
  -Db_sanitize=address \
  -Dbuildtype=debugoptimized \
  -Dtests=true
```

**Environment Variables:**
```bash
# Enable leak detection (default with ASAN)
export ASAN_OPTIONS=detect_leaks=1:symbolize=1:abort_on_error=1

# Disable leak detection if needed (e.g., for known leaks)
export ASAN_OPTIONS=detect_leaks=0

# Run tests
meson test -C builddir-asan
```

**Example Output:**
```
=================================================================
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000013 at pc 0x000000400e3c bp 0x7fffffffd8f0 sp 0x7fffffffd8e8
READ of size 1 at 0x602000000013 thread T0
    #0 0x400e3b in fvec_quadratic_peak_mag src/mathutils.c:507
    #1 0x400f2d in aubio_beattracking_get_confidence src/tempo/beattracking.c:445
```

### UndefinedBehaviorSanitizer (UBSAN)

**Detects:**
- Integer overflow
- Division by zero
- Invalid shifts
- Out-of-bounds array access
- Type mismatches
- Null pointer dereference
- Misaligned pointers

**Setup:**
```bash
meson setup builddir-ubsan \
  -Db_sanitize=undefined \
  -Dbuildtype=debugoptimized \
  -Dtests=true
```

**Environment Variables:**
```bash
# Print full stack traces
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

# Run tests
meson test -C builddir-ubsan
```

**Example Output:**
```
src/mathutils.c:502:31: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'
```

### MemorySanitizer (MSAN)

**Detects:**
- Use of uninitialized memory

**Note:** MSAN requires all libraries to be instrumented, making it complex to use. See advanced section.

**Setup:**
```bash
# Requires Clang
CC=clang meson setup builddir-msan \
  -Db_sanitize=memory \
  -Dbuildtype=debugoptimized \
  -Dtests=true
```

### ThreadSanitizer (TSAN)

**Detects:**
- Data races in multithreaded code

**Note:** aubio is primarily single-threaded, but TSAN is useful if using aubio in a multithreaded application.

**Setup:**
```bash
meson setup builddir-tsan \
  -Db_sanitize=thread \
  -Dbuildtype=debugoptimized \
  -Dtests=true
```

## Performance Impact

| Sanitizer | Slowdown | Memory Overhead |
|-----------|----------|-----------------|
| ASAN | 2x | 2-3x |
| UBSAN | 1.2x | Minimal |
| MSAN | 3x | 2x |
| TSAN | 5-15x | 5-10x |

**Recommendation:** Use ASAN+UBSAN for regular testing, as the overhead is acceptable.

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/sanitizers.yml`:

```yaml
name: Sanitizer Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  asan-ubsan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y meson ninja-build python3-pip
          pip install numpy
      
      - name: Setup build with sanitizers
        run: |
          meson setup builddir \
            -Db_sanitize=address,undefined \
            -Dbuildtype=debugoptimized \
            -Dtests=true
      
      - name: Build
        run: meson compile -C builddir
      
      - name: Run tests with sanitizers
        run: meson test -C builddir --print-errorlogs
        env:
          ASAN_OPTIONS: detect_leaks=1:symbolize=1:abort_on_error=1
          UBSAN_OPTIONS: print_stacktrace=1:halt_on_error=1
      
      - name: Upload test logs on failure
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: sanitizer-test-logs
          path: builddir/meson-logs/
```

## Interpreting Results

### ASAN Reports

**Heap Buffer Overflow Example:**
```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow
READ of size 4 at 0x6020000000bc
    #0 0x4008e3 in vulnerable_function src/file.c:42
```

**Action:** Check array access at line 42, ensure bounds are validated.

**Use-After-Free Example:**
```
==12345==ERROR: AddressSanitizer: heap-use-after-free
READ of size 4 at 0x6020000000bc
    #0 0x4008e3 in bad_function src/file.c:55
```

**Action:** Object was freed before use. Check object lifecycle.

### UBSAN Reports

**Integer Overflow Example:**
```
src/file.c:23:15: runtime error: signed integer overflow: 2147483647 + 1
```

**Action:** Use unsigned integers or add overflow checks.

**Division by Zero Example:**
```
src/file.c:45:10: runtime error: division by zero
```

**Action:** Add zero check before division.

## Common Issues and Solutions

### Issue: Tests fail with sanitizers but pass without

**Cause:** Real bugs that were previously undetected.

**Solution:** Fix the reported issues - they are likely genuine bugs.

### Issue: False positive in system libraries

**Cause:** ASAN detects issues in dependencies not compiled with sanitizers.

**Solution:** Use suppression file:

Create `asan.supp`:
```
# Suppress known issue in system library
leak:libsystemlibrary.so
```

Use it:
```bash
export LSAN_OPTIONS=suppressions=asan.supp
meson test -C builddir-asan
```

### Issue: Performance is too slow

**Cause:** Sanitizers add significant overhead.

**Solution:** 
- Use sanitizers only during development/testing
- Run subset of tests with sanitizers
- Use UBSAN only (lowest overhead)

### Issue: Out of memory

**Cause:** ASAN has high memory overhead.

**Solution:**
- Increase system memory/swap
- Run tests individually
- Use smaller test data sets

## Best Practices

### 1. Regular Testing

Run sanitizer tests:
- Before committing changes
- In CI/CD pipeline
- Weekly on main branch

### 2. Fix Issues Immediately

Don't ignore sanitizer warnings - they indicate real bugs.

### 3. Test with Different Build Types

```bash
# Debug build
meson setup builddir-debug -Dbuildtype=debug -Db_sanitize=address,undefined

# Release build with sanitizers (catches optimization-related issues)
meson setup builddir-release -Dbuildtype=release -Db_sanitize=address,undefined
```

### 4. Combine Multiple Sanitizers

ASAN + UBSAN is the recommended combination:
```bash
meson setup builddir -Db_sanitize=address,undefined -Dtests=true
```

### 5. Use Environment Variables

```bash
# Comprehensive ASAN options
export ASAN_OPTIONS=\
detect_leaks=1:\
symbolize=1:\
abort_on_error=1:\
detect_stack_use_after_return=1:\
check_initialization_order=1:\
strict_init_order=1

# Comprehensive UBSAN options
export UBSAN_OPTIONS=\
print_stacktrace=1:\
halt_on_error=1:\
report_error_type=1
```

## Advanced Usage

### Custom Test Scripts

Create `scripts/test-sanitizers.sh`:

```bash
#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Setup ASAN environment
export ASAN_OPTIONS=detect_leaks=1:symbolize=1:abort_on_error=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1

# Build with sanitizers
echo "Building with sanitizers..."
meson setup builddir-sanitizers \
  -Db_sanitize=address,undefined \
  -Dbuildtype=debugoptimized \
  -Dtests=true

meson compile -C builddir-sanitizers

# Run tests
echo "Running tests with sanitizers..."
if meson test -C builddir-sanitizers --print-errorlogs; then
  echo -e "${GREEN}All sanitizer tests passed!${NC}"
  exit 0
else
  echo -e "${RED}Sanitizer tests failed!${NC}"
  cat builddir-sanitizers/meson-logs/testlog.txt
  exit 1
fi
```

### Valgrind Integration

For additional checking, combine with Valgrind:

```bash
# Build without sanitizers (Valgrind doesn't work well with ASAN)
meson setup builddir-valgrind -Dtests=true -Dbuildtype=debugoptimized

# Run with Valgrind
meson test -C builddir-valgrind --wrap="valgrind --leak-check=full --show-leak-kinds=all"
```

## Sanitizer Compatibility

| Sanitizer | GCC | Clang | MSVC |
|-----------|-----|-------|------|
| ASAN | ✅ 4.8+ | ✅ 3.1+ | ✅ 2019+ |
| UBSAN | ✅ 4.9+ | ✅ 3.3+ | ❌ |
| MSAN | ❌ | ✅ 3.3+ | ❌ |
| TSAN | ✅ 4.8+ | ✅ 3.2+ | ❌ |

## References

- [AddressSanitizer Documentation](https://github.com/google/sanitizers/wiki/AddressSanitizer)
- [UndefinedBehaviorSanitizer Documentation](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
- [Meson Sanitizer Support](https://mesonbuild.com/Builtin-options.html#details-for-b_sanitize)
- [Google Sanitizers Wiki](https://github.com/google/sanitizers/wiki)

## Troubleshooting

### Sanitizer not found

**Error:** `meson.build:1:0: ERROR: Compiler does not support sanitizer`

**Solution:** 
```bash
# Check compiler version
gcc --version  # Need GCC 4.8+ for ASAN
clang --version  # Need Clang 3.1+ for ASAN

# Try with Clang if GCC is old
CC=clang meson setup builddir -Db_sanitize=address
```

### Symbol not found

**Error:** `undefined reference to __asan_init`

**Solution:** Ensure proper linking:
```bash
# Clean rebuild
rm -rf builddir
meson setup builddir -Db_sanitize=address
```

### Python extension fails to load

**Error:** `ImportError: ASAN runtime not found`

**Solution:** Set LD_PRELOAD:
```bash
# Find ASAN library
ASAN_LIB=$(gcc -print-file-name=libasan.so)

# Preload it
LD_PRELOAD=$ASAN_LIB python -c "import aubio"
```

---

**Last Updated:** 2025-11-14  
**Version:** 0.5.0-alpha
