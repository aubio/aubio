# Defensive Programming in aubio-ledfx

This document describes the defensive programming patterns and security assertions implemented in aubio-ledfx to prevent common programming errors and security vulnerabilities.

## Overview

Defensive programming is a practice of writing code that anticipates and handles potential errors gracefully. In aubio-ledfx, we use compile-time checks, runtime assertions, and input validation to ensure code correctness and security.

## Security Assertion Macros

### Available Macros

All security assertion macros are defined in `src/aubio_priv.h` and are active in:
- **Debug builds** (`-Dbuildtype=debug`)
- **Builds with security checks enabled** (`-DAUBIO_SECURITY_CHECKS`)

In release builds, these macros are no-ops for performance.

#### AUBIO_ASSERT_BOUNDS(idx, len)

**Purpose:** Validates that an array index is within valid range.

**When to use:**
- Before accessing array elements
- Before pointer arithmetic
- In loops that access arrays

**Example:**
```c
void process_data(fvec_t *vec, uint_t index) {
  AUBIO_ASSERT_BOUNDS(index, vec->length);
  vec->data[index] = 0.0;
}
```

**Behavior:**
- **Debug:** Aborts with error message if `idx >= len`
- **Release:** No-op (compiled out)

#### AUBIO_ASSERT_NOT_NULL(ptr)

**Purpose:** Validates that a pointer is not NULL.

**When to use:**
- At the start of public API functions
- Before dereferencing pointers
- After allocations (though allocation failures are usually fatal)

**Example:**
```c
smpl_t fvec_get_sample(const fvec_t *s, uint_t position) {
  AUBIO_ASSERT_NOT_NULL(s);
  AUBIO_ASSERT_BOUNDS(position, s->length);
  return s->data[position];
}
```

**Behavior:**
- **Debug:** Aborts with error message if pointer is NULL
- **Release:** No-op (compiled out)

#### AUBIO_ASSERT_RANGE(val, min, max)

**Purpose:** Validates that a value is within an expected range.

**When to use:**
- For normalized values (e.g., must be in [0, 1])
- For frequency values that must be positive
- For parameters with known valid ranges

**Example:**
```c
void set_threshold(aubio_onset_t *o, smpl_t threshold) {
  AUBIO_ASSERT_NOT_NULL(o);
  AUBIO_ASSERT_RANGE(threshold, 0.0, 1.0);
  o->threshold = threshold;
}
```

**Behavior:**
- **Debug:** Aborts if value is outside [min, max]
- **Release:** No-op (compiled out)

#### AUBIO_ASSERT_LENGTH(buf, expected)

**Purpose:** Validates that a buffer has the expected length.

**When to use:**
- When functions expect buffers of specific sizes
- Before operations that assume matching buffer sizes
- In conjunction with input validation

**Example:**
```c
void process_fft(aubio_fft_t *fft, fvec_t *input, cvec_t *output) {
  AUBIO_ASSERT_NOT_NULL(fft);
  AUBIO_ASSERT_NOT_NULL(input);
  AUBIO_ASSERT_NOT_NULL(output);
  AUBIO_ASSERT_LENGTH(input, fft->winsize);
  AUBIO_ASSERT_LENGTH(output, fft->winsize / 2 + 1);
  // ... perform FFT
}
```

**Behavior:**
- **Debug:** Aborts if buffer length doesn't match expected
- **Release:** No-op (compiled out)

## Input Validation Patterns

### Public API Functions

**All public API functions should validate their inputs:**

```c
// Good: Validates inputs
void aubio_func(fvec_t *input, fvec_t *output) {
  // 1. Check for NULL pointers
  if (!input || !output) {
    AUBIO_ERR("aubio_func: null input\n");
    return;
  }
  
  // 2. Validate buffer sizes
  if (input->length == 0 || output->length == 0) {
    AUBIO_WRN("aubio_func: empty buffer\n");
    return;
  }
  
  // 3. In debug mode, use assertions for extra checking
  AUBIO_ASSERT_NOT_NULL(input);
  AUBIO_ASSERT_NOT_NULL(output);
  
  // 4. Proceed with actual work
  // ...
}
```

### Array Access Pattern

**Always validate before accessing arrays:**

```c
// Good: Safe array access
void safe_array_access(fvec_t *vec, uint_t index) {
  // Runtime check (always active)
  if (index >= vec->length) {
    AUBIO_ERR("index out of bounds: %u >= %u\n", index, vec->length);
    return;
  }
  
  // Debug assertion (only in debug builds)
  AUBIO_ASSERT_BOUNDS(index, vec->length);
  
  // Safe to access now
  vec->data[index] = 0.0;
}
```

### Loop Bounds Pattern

**Ensure loops don't go out of bounds:**

```c
// Good: Safe loop with explicit bounds
void process_vector(fvec_t *vec) {
  uint_t i;
  AUBIO_ASSERT_NOT_NULL(vec);
  
  for (i = 0; i < vec->length; i++) {
    // In debug builds, this adds extra validation
    AUBIO_ASSERT_BOUNDS(i, vec->length);
    vec->data[i] = process_sample(vec->data[i]);
  }
}

// Good: Safe access with lookahead
void process_with_lookahead(fvec_t *vec) {
  uint_t i;
  AUBIO_ASSERT_NOT_NULL(vec);
  
  // Loop stops early to prevent out-of-bounds access
  for (i = 0; i < vec->length - 1; i++) {
    AUBIO_ASSERT_BOUNDS(i + 1, vec->length);
    vec->data[i] = (vec->data[i] + vec->data[i + 1]) / 2.0;
  }
}
```

## Build Configuration

### Debug Build (Assertions Enabled)

```bash
# Enable all security assertions
meson setup builddir -Dbuildtype=debug -Dtests=true
meson compile -C builddir
meson test -C builddir
```

### Release Build (Assertions Disabled)

```bash
# Disable assertions for performance
meson setup builddir -Dbuildtype=release -Dtests=true
meson compile -C builddir
```

### Release Build with Security Checks

```bash
# Keep assertions in release for extra safety
meson setup builddir \
  -Dbuildtype=release \
  -Dc_args='-DAUBIO_SECURITY_CHECKS' \
  -Dtests=true
```

## Testing Assertions

### Manual Testing

A test program is provided to verify assertions work:

```bash
# Build in debug mode
meson setup builddir -Dbuildtype=debug -Dtests=true
meson compile -C builddir

# Test bounds checking (will abort)
./builddir/tests/test-security-assertions bounds

# Test null pointer checking (will abort)
./builddir/tests/test-security-assertions null
```

**Expected behavior:** Program should abort with error message.

### Automated Testing

Include edge case tests that would trigger assertions:

```c
int test_edge_cases(void) {
  fvec_t *vec = new_fvec(10);
  
  // Test valid boundary
  fvec_set_sample(vec, 1.0, 9);  // OK: last valid index
  assert(fvec_get_sample(vec, 9) == 1.0);
  
  // In release builds, this may cause undefined behavior
  // In debug builds, this will abort with assertion
  // Don't include in regular test suite!
  
  del_fvec(vec);
  return 0;
}
```

## Best Practices

### 1. Validate Early, Validate Often

```c
// Good: Validate at function entry
void process(fvec_t *in, fvec_t *out, uint_t length) {
  // Validate inputs first
  AUBIO_ASSERT_NOT_NULL(in);
  AUBIO_ASSERT_NOT_NULL(out);
  AUBIO_ASSERT_LENGTH(in, length);
  AUBIO_ASSERT_LENGTH(out, length);
  
  // Proceed with confidence
  for (uint_t i = 0; i < length; i++) {
    out->data[i] = in->data[i] * 2.0;
  }
}
```

### 2. Use Assertions for Preconditions

```c
// Good: Document and check preconditions
/**
 * Process audio with FFT
 * 
 * @pre input must have length equal to fft->winsize
 * @pre output must have length equal to fft->winsize / 2 + 1
 */
void aubio_fft_do(aubio_fft_t *fft, fvec_t *input, cvec_t *output) {
  AUBIO_ASSERT_NOT_NULL(fft);
  AUBIO_ASSERT_NOT_NULL(input);
  AUBIO_ASSERT_NOT_NULL(output);
  AUBIO_ASSERT_LENGTH(input, fft->winsize);
  // ... perform FFT
}
```

### 3. Combine with Runtime Checks

```c
// Good: Assertions for development, checks for production
void safe_function(fvec_t *vec, uint_t index) {
  // Runtime check (always active)
  if (!vec || index >= vec->length) {
    AUBIO_ERR("invalid input\n");
    return;
  }
  
  // Debug assertion (helps during development)
  AUBIO_ASSERT_NOT_NULL(vec);
  AUBIO_ASSERT_BOUNDS(index, vec->length);
  
  // Safe to proceed
  vec->data[index] = 0.0;
}
```

### 4. Don't Use Assertions for Error Handling

```c
// Bad: Assertions for user errors
void bad_function(const char *filename) {
  AUBIO_ASSERT_NOT_NULL(filename);  // Wrong!
  FILE *f = fopen(filename, "r");
  AUBIO_ASSERT_NOT_NULL(f);  // Wrong! User error, not programmer error
}

// Good: Proper error handling
void good_function(const char *filename) {
  if (!filename) {
    AUBIO_ERR("filename is null\n");
    return;
  }
  FILE *f = fopen(filename, "r");
  if (!f) {
    AUBIO_ERR("cannot open file: %s\n", filename);
    return;
  }
  // ... use file
}
```

## Performance Considerations

### Assertion Overhead

In debug builds:
- Assertions add minimal overhead (~1-5%)
- Essential for development and testing
- Help catch bugs early

In release builds:
- Assertions are compiled out
- Zero runtime overhead
- Still have runtime error checks for critical cases

### When to Keep Assertions in Release

Consider keeping assertions enabled (`-DAUBIO_SECURITY_CHECKS`) when:
- Running in security-critical environments
- Processing untrusted input
- The <5% performance cost is acceptable
- Extra safety is worth the trade-off

## Common Patterns in aubio-ledfx

### Vector Operations

```c
void fvec_set_sample(fvec_t *s, smpl_t data, uint_t position) {
  AUBIO_ASSERT_NOT_NULL(s);
  AUBIO_ASSERT_BOUNDS(position, s->length);
  s->data[position] = data;
}
```

### Object Initialization

```c
aubio_obj_t *new_aubio_obj(uint_t size) {
  aubio_obj_t *o = AUBIO_NEW(aubio_obj_t);
  if (!o) {
    AUBIO_ERR("failed to allocate object\n");
    return NULL;
  }
  
  o->buffer = new_fvec(size);
  if (!o->buffer) {
    AUBIO_ERR("failed to allocate buffer\n");
    AUBIO_FREE(o);
    return NULL;
  }
  
  AUBIO_ASSERT_NOT_NULL(o->buffer);
  return o;
}
```

### Processing Functions

```c
void aubio_process(aubio_obj_t *o, fvec_t *in, fvec_t *out) {
  AUBIO_ASSERT_NOT_NULL(o);
  AUBIO_ASSERT_NOT_NULL(in);
  AUBIO_ASSERT_NOT_NULL(out);
  AUBIO_ASSERT_LENGTH(in, o->input_size);
  AUBIO_ASSERT_LENGTH(out, o->output_size);
  
  // Safe to process
  for (uint_t i = 0; i < in->length; i++) {
    out->data[i] = in->data[i] * o->gain;
  }
}
```

## Integration with Other Security Features

Defensive programming works best when combined with:

1. **Compiler Security Flags** (see `SECURITY_HARDENING.md`)
   - Stack protection
   - Fortify source
   - Format security

2. **Sanitizers** (see `SANITIZERS.md`)
   - AddressSanitizer
   - UndefinedBehaviorSanitizer
   - Memory sanitizer

3. **Static Analysis** (planned)
   - Clang-Tidy
   - Cppcheck
   - CodeQL

## References

- [Secure Coding in C](https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+C+Coding+Standard)
- [Defensive Programming](https://en.wikipedia.org/wiki/Defensive_programming)
- [MISRA C Guidelines](https://www.misra.org.uk/)

---

**Last Updated:** 2025-11-14  
**Version:** 0.5.0-alpha
