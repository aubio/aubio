# Security Hardening in aubio-ledfx

This document describes the security hardening measures implemented in the aubio-ledfx build system.

## Overview

Security hardening consists of compiler and linker flags that add runtime protections and compile-time checks to help prevent common vulnerabilities like buffer overflows, format string attacks, and other memory corruption issues.

## Enabled Security Features

### 1. Stack Protection (`-fstack-protector-strong`)

**What it does:** Adds canary values to the stack to detect buffer overflows before they can be exploited.

**Coverage:** Functions with:
- Local arrays
- References to local frame addresses
- Calls to `alloca()`
- Register spills

**Overhead:** Minimal (<1% performance impact)

**Detection:** Runtime check; program aborts with "stack smashing detected" message if overflow is detected.

### 2. Fortify Source (`-D_FORTIFY_SOURCE=2`)

**What it does:** Replaces standard library functions (strcpy, memcpy, etc.) with bounds-checking versions that validate buffer sizes at compile-time and runtime.

**Requirements:** Only active in optimized builds (`-O1` or higher)

**Protection against:**
- Buffer overflows in string operations
- Format string vulnerabilities
- Memory copy operations exceeding buffer bounds

**Example:** `strcpy(dest, src)` becomes `__strcpy_chk(dest, src, sizeof(dest))`

### 3. Format String Security (`-Wformat -Wformat-security`)

**What it does:** 
- `-Wformat`: Warns about incorrect format strings
- `-Wformat-security`: Warns when format strings are not string literals (potential security issue)

**Prevents:**
- Format string vulnerabilities where attacker-controlled input is used as format string
- Type mismatches in printf-family functions

**Example Warning:**
```c
char *input = get_user_input();
printf(input);  // WARNING: format not a string literal
```

### 4. Implicit Function Declaration Errors (`-Werror=implicit-function-declaration`)

**What it does:** Treats implicit function declarations as errors instead of warnings.

**Why it matters:**
- Prevents calling functions with wrong signatures
- Ensures proper type checking
- Avoids undefined behavior from incorrect return type assumptions

**Example:**
```c
int main() {
    result = strlen("test");  // ERROR: strlen not declared
    return 0;
}
```

### 5. Array Bounds Checking (`-Warray-bounds`)

**What it does:** Warns about array subscripts that are always out of bounds or likely to be out of bounds.

**Protection:**
- Compile-time detection of obvious out-of-bounds accesses
- Helps catch indexing errors before runtime

### 6. Position Independent Executables (PIE)

**What it does:** Creates executables with randomized memory addresses (ASLR - Address Space Layout Randomization).

**Security benefit:**
- Makes exploits harder by randomizing memory layout
- Prevents hardcoded addresses in exploits

**How to enable:**
```bash
meson setup builddir -Db_pie=true
```

**Note:** PIE is automatically enabled for shared libraries via `-fPIC`.

## Build Configuration

### Enable Security Hardening (Default)

```bash
meson setup builddir
# or explicitly:
meson setup builddir -Dsecurity_hardening=true
```

### Disable Security Hardening (For Compatibility)

```bash
meson setup builddir -Dsecurity_hardening=false
```

**Note:** Disabling is not recommended for production builds.

### Recommended Build for Maximum Security

```bash
meson setup builddir \
  -Dbuildtype=release \
  -Dsecurity_hardening=true \
  -Db_pie=true \
  -Dtests=true

meson compile -C builddir
meson test -C builddir
```

## Performance Impact

| Feature | Performance Impact | Memory Impact |
|---------|-------------------|---------------|
| Stack Protector | <1% | +8 bytes per protected function |
| Fortify Source | <2% | Minimal |
| Format Warnings | None (compile-time) | None |
| PIE | <1% | None |
| **Total** | **<3%** | **Minimal** |

The security benefits far outweigh the minimal performance cost.

## Compiler Support

All features are supported by:
- GCC 4.9+
- Clang 3.4+
- MSVC 2015+ (partial support)

The build system automatically detects compiler support and enables only the features available.

## Security Features by Build Type

| Build Type | Fortify Source | Stack Protector | Other Flags |
|------------|----------------|-----------------|-------------|
| `debug` | ❌ (requires optimization) | ✅ | ✅ |
| `debugoptimized` | ✅ | ✅ | ✅ |
| `release` | ✅ | ✅ | ✅ |
| `minsize` | ✅ | ✅ | ✅ |
| `plain` | ❌ | ✅ | ✅ |

## Testing Security Features

### 1. Verify Stack Protection

Create a test program with buffer overflow:

```c
#include <string.h>

void vulnerable() {
    char buffer[8];
    strcpy(buffer, "This string is way too long");
}

int main() {
    vulnerable();
    return 0;
}
```

Compile and run:
```bash
gcc test.c -fstack-protector-strong -o test
./test
# Should output: *** stack smashing detected ***
```

### 2. Verify Fortify Source

Create test with buffer overflow:

```c
#include <string.h>

int main() {
    char dest[8];
    char src[20] = "This is too long";
    strcpy(dest, src);  // Will be detected
    return 0;
}
```

Compile and run:
```bash
gcc test.c -O2 -D_FORTIFY_SOURCE=2 -o test
./test
# Should abort with buffer overflow detected
```

### 3. Check Enabled Flags

View actual compile commands:
```bash
meson compile -C builddir -v 2>&1 | grep "fstack-protector"
```

## Integration with CI/CD

Security hardening is automatically enabled in the GitHub Actions workflows. To ensure it's active:

```yaml
- name: Build with security hardening
  run: |
    meson setup builddir -Dsecurity_hardening=true -Db_pie=true
    meson compile -C builddir
    meson test -C builddir
```

## Limitations and Considerations

### Platform-Specific Behavior

1. **Windows:** Some features (like PIE) work differently on Windows due to ASLR being handled by the OS.

2. **macOS:** Stack protection and PIE are enabled by default in recent XCode versions.

3. **Embedded Systems:** Some flags may not be available or may have higher overhead.

### Known Issues

1. **Fortify Source and Debug Builds:** `-D_FORTIFY_SOURCE=2` requires optimization and is disabled in debug builds.

2. **False Positives:** `-Warray-bounds` may occasionally warn about safe code. Use `#pragma GCC diagnostic` to suppress specific warnings if needed.

3. **PIE Overhead:** On very old systems (pre-2010), PIE may have 1-3% overhead. On modern systems, overhead is negligible.

## Additional Security Measures

Beyond compiler flags, aubio-ledfx implements:

1. **Explicit Bounds Checking:** All array accesses are validated (see `SECURITY_REVIEW.md`)

2. **Safe String Handling:** No use of unsafe functions like `strcpy`, `sprintf`, `gets`

3. **Input Validation:** All public API functions validate input parameters

4. **Memory Safety:** Proper allocation sizing and null termination for strings

5. **CodeQL Analysis:** Automated security scanning in CI/CD

## References

- [Debian Hardening Guide](https://wiki.debian.org/Hardening)
- [GCC Security Options](https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [Linux Kernel Self Protection](https://www.kernel.org/doc/html/latest/security/self-protection.html)

## Future Enhancements

Planned security improvements:

- [ ] Enable Control Flow Integrity (CFI) when widely supported
- [ ] Add integer overflow protection (`-ftrapv`)
- [ ] Implement shadow stack (when available)
- [ ] Add memory sanitizer builds for CI
- [ ] Enable link-time optimization (LTO) for better optimization + security

## Contact

For security-related questions or to report vulnerabilities, please see `SECURITY.md`.

---

**Last Updated:** 2025-11-14  
**Version:** 0.5.0-alpha
