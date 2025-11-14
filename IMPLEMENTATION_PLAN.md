# Security & Stability Strengthening - Implementation Plan

**Project:** aubio-ledfx  
**Created:** 2025-11-14  
**Based on:** SECURITY_REVIEW.md and FUTURE_ACTIONS.md  
**Goal:** Systematic security hardening and stability improvements for the aubio codebase

---

## Executive Summary

This plan addresses security and stability improvements identified in the comprehensive security review. While 4 critical vulnerabilities have already been fixed, this plan focuses on:

1. **Remaining Security Items** - Address low-priority security verification needs
2. **Testing Infrastructure** - Add comprehensive security testing (sanitizers, fuzz testing)
3. **Code Quality** - Implement defensive programming patterns and static analysis
4. **CI/CD Integration** - Automate security checks in development workflow
5. **Documentation** - Provide security guidelines for contributors

**Current State:** All 45 tests passing ✅, 4 critical vulnerabilities fixed ✅

---

## Phase 1: Immediate Security Verification (Priority: HIGH)

### 1.1 fvec_quadratic_peak_mag Bounds Validation

**Status:** 🔄 IN PROGRESS  
**Priority:** HIGH (security-related, low effort)  
**Effort:** 2-4 hours  
**Files:** `src/mathutils.c`, `src/tempo/beattracking.c`

**Tasks:**
- [ ] Add explicit bounds check: `if (index + 1 >= x->length) return x->data[index];`
- [ ] Audit the single call site in `beattracking.c:445`
- [ ] Verify `bt->gp` can never cause out-of-bounds access
- [ ] Add unit test for peak at boundary position
- [ ] Document preconditions in function comment

**Rationale:** Currently flagged in FUTURE_ACTIONS.md as requiring verification. The bounds check `pos >= x->length` doesn't guarantee `index + 1 < x->length` due to integer conversion formula.

---

## Phase 2: Build Hardening & Compile-Time Security (Priority: HIGH)

### 2.1 Add Security Compiler Flags

**Status:** 🔄 IN PROGRESS  
**Priority:** HIGH  
**Effort:** 4 hours  
**Files:** `meson.build`

**Tasks:**
- [ ] Add `-D_FORTIFY_SOURCE=2` for buffer overflow detection
- [ ] Enable `-fstack-protector-strong` for stack protection
- [ ] Add `-Wformat -Wformat-security` for format string checking
- [ ] Enable `-Werror=implicit-function-declaration` to catch undeclared functions
- [ ] Add `-fPIE -pie` for position-independent executables
- [ ] Document security flags in build configuration

**Implementation:**
```meson
# Security hardening flags
security_flags = [
  '-D_FORTIFY_SOURCE=2',
  '-fstack-protector-strong',
  '-Wformat',
  '-Wformat-security',
  '-Werror=implicit-function-declaration',
]

if get_option('buildtype') == 'release'
  security_flags += ['-fPIE']
endif
```

### 2.2 Debug Build Assertions

**Status:** 🔄 IN PROGRESS  
**Priority:** MEDIUM  
**Effort:** 6 hours  
**Files:** `src/aubio_priv.h`, various source files

**Tasks:**
- [ ] Create `AUBIO_ASSERT_BOUNDS(index, length)` macro
- [ ] Add assertions for all array accesses in debug builds
- [ ] Create `AUBIO_CHECKED_ACCESS(array, index)` wrapper
- [ ] Add static assertions for compile-time size validation
- [ ] Document assertion macros in developer guide

**Implementation:**
```c
#ifdef NDEBUG
  #define AUBIO_ASSERT_BOUNDS(idx, len) ((void)0)
#else
  #define AUBIO_ASSERT_BOUNDS(idx, len) \
    do { if ((idx) >= (len)) { \
      AUBIO_ERR("bounds check failed: %u >= %u at %s:%d", \
                (idx), (len), __FILE__, __LINE__); \
      abort(); \
    }} while(0)
#endif
```

---

## Phase 3: Memory Safety Testing (Priority: HIGH)

### 3.1 AddressSanitizer (ASAN) Integration

**Status:** 🔄 IN PROGRESS  
**Priority:** HIGH  
**Effort:** 1 day  
**Files:** `meson_options.txt`, `meson.build`, `.github/workflows/build.yml`

**Tasks:**
- [ ] Add `sanitizers` option to meson_options.txt
- [ ] Create sanitizer build configuration
- [ ] Add ASAN test job to CI workflow
- [ ] Run full test suite with ASAN enabled
- [ ] Fix any issues detected by ASAN
- [ ] Document ASAN usage in developer guide

**Implementation:**
```meson
# meson_options.txt
option('sanitizers',
  type: 'array',
  choices: ['address', 'undefined', 'memory', 'thread'],
  value: [],
  description: 'Enable sanitizers for security testing'
)
```

### 3.2 Valgrind Integration

**Status:** 🔄 IN PROGRESS  
**Priority:** MEDIUM  
**Effort:** 4 hours  
**Files:** `meson.build`, test scripts

**Tasks:**
- [ ] Add Valgrind test target
- [ ] Create suppression file for known false positives
- [ ] Run test suite under Valgrind memcheck
- [ ] Fix any memory leaks detected
- [ ] Add Valgrind CI job
- [ ] Document Valgrind workflow

**Implementation:**
```bash
# Add to test script
valgrind --leak-check=full --show-leak-kinds=all \
         --error-exitcode=1 --suppressions=valgrind.supp \
         ./builddir/tests/test-*
```

### 3.3 UndefinedBehaviorSanitizer (UBSAN)

**Status:** ⏸️ PLANNED  
**Priority:** MEDIUM  
**Effort:** 4 hours  
**Files:** `meson.build`, CI workflow

**Tasks:**
- [ ] Enable UBSAN in sanitizer builds
- [ ] Run test suite with UBSAN
- [ ] Fix any undefined behavior detected
- [ ] Add UBSAN to CI pipeline

---

## Phase 4: Comprehensive Test Coverage (Priority: MEDIUM)

### 4.1 Boundary Condition Tests

**Status:** 🔄 IN PROGRESS  
**Priority:** MEDIUM  
**Effort:** 2-3 days  
**Files:** `tests/src/spectral/`, `tests/src/pitch/`, `tests/src/tempo/`

**Test Cases to Add:**

**Spectral Analysis:**
- [ ] Test spectral rolloff with energy in first bin
- [ ] Test spectral rolloff with energy in last bin
- [ ] Test spectral rolloff with energy in middle bin
- [ ] Test spectral rolloff with distributed energy
- [ ] Test spectral descriptors with zero input
- [ ] Test spectral descriptors with single spike

**Pitch Detection:**
- [ ] Test pitch detection with DC signal
- [ ] Test pitch detection at Nyquist frequency
- [ ] Test pitch detection with silence
- [ ] Test Schmitt trigger at buffer boundaries
- [ ] Test pitch algorithms with extreme frequencies

**Beat Tracking:**
- [ ] Test very slow tempo (<30 BPM)
- [ ] Test very fast tempo (>300 BPM)
- [ ] Test tempo with irregular beats
- [ ] Test beattracking peak detection at boundaries

**Buffer Operations:**
- [ ] Test with length 1 buffers
- [ ] Test with maximum size buffers
- [ ] Test with empty buffers (length 0)
- [ ] Test all shifting operations at boundaries

**Sample Rates:**
- [ ] Test at 8kHz, 96kHz, 192kHz
- [ ] Test with mismatched sample rates
- [ ] Test resampling edge cases

### 4.2 Fuzz Testing Infrastructure

**Status:** ⏸️ PLANNED  
**Priority:** MEDIUM  
**Effort:** 1-2 days  
**Files:** `fuzz/`, CI workflow

**Tasks:**
- [ ] Set up libFuzzer or AFL++ framework
- [ ] Create harness for spectral analysis functions
- [ ] Create harness for pitch detection algorithms
- [ ] Create harness for tempo/beat tracking
- [ ] Create harness for audio I/O with malformed files
- [ ] Run fuzzer for 24+ hours initially
- [ ] Add regression tests for any crashes found
- [ ] Integrate fuzz testing in CI (short runs)

**Harness Example:**
```c
// fuzz/fuzz_specdesc.c
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < sizeof(uint_t) * 2) return 0;
  
  uint_t win_s = *(uint_t*)data;
  if (win_s < 4 || win_s > 8192) return 0;
  
  aubio_specdesc_t *o = new_aubio_specdesc("rolloff", win_s);
  cvec_t *in = new_cvec(win_s);
  fvec_t *out = new_fvec(1);
  
  // Fuzz the input spectrum
  for (uint_t i = 0; i < in->length && i < size; i++) {
    in->norm[i] = ((smpl_t)data[i]) / 255.0;
  }
  
  aubio_specdesc_do(o, in, out);
  
  del_fvec(out);
  del_cvec(in);
  del_aubio_specdesc(o);
  return 0;
}
```

### 4.3 Memory Leak Detection

**Status:** 🔄 IN PROGRESS  
**Priority:** HIGH  
**Effort:** 1 day  
**Files:** Test suite

**Tasks:**
- [ ] Run test suite with LeakSanitizer
- [ ] Fix any memory leaks found
- [ ] Add memory leak checks to CI
- [ ] Create memory management guidelines

---

## Phase 5: Static Analysis & Code Quality (Priority: MEDIUM)

### 5.1 Clang-Tidy Integration

**Status:** 🔄 IN PROGRESS  
**Priority:** MEDIUM  
**Effort:** 2-3 days  
**Files:** `.clang-tidy`, meson.build, CI workflow

**Tasks:**
- [ ] Create .clang-tidy configuration
- [ ] Enable array-bounds checks
- [ ] Enable buffer-overflow checks
- [ ] Enable security-focused checks
- [ ] Fix all issues found
- [ ] Add to CI pipeline as required check
- [ ] Add to pre-commit hooks

**Configuration:**
```yaml
# .clang-tidy
Checks: >
  clang-analyzer-*,
  -clang-analyzer-security.insecureAPI.strcpy,
  cppcoreguidelines-*,
  readability-*,
  modernize-*,
  performance-*,
  bugprone-*
WarningsAsErrors: '*'
```

### 5.2 Additional Static Analysis Tools

**Status:** ⏸️ PLANNED  
**Priority:** LOW  
**Effort:** 2-3 days per tool  

**Tools to Add:**
- [ ] **Cppcheck** - Add to pre-commit hooks
- [ ] **Coverity Scan** - Register for free OSS scanning
- [ ] **Infer** - Run on CI for high-confidence issues
- [ ] **SonarQube** - For comprehensive code quality metrics

### 5.3 Defensive Programming Patterns

**Status:** 🔄 IN PROGRESS  
**Priority:** MEDIUM  
**Effort:** 3-5 days  
**Files:** Various source files

**Tasks:**
- [ ] Add NULL pointer checks to all public API functions
- [ ] Add input validation for all parameters
- [ ] Replace manual pointer arithmetic with array indexing
- [ ] Create `fvec_get`/`fvec_set` wrappers with bounds checking
- [ ] Add debug-mode assertions for all array accesses
- [ ] Use `static_assert` for compile-time size validation

**Example Pattern:**
```c
// Before:
void aubio_func(fvec_t *vec) {
  for (i = 0; i < vec->length; i++) {
    vec->data[i] = ...;
  }
}

// After:
void aubio_func(fvec_t *vec) {
  if (!vec || !vec->data) {
    AUBIO_ERR("aubio_func: null input");
    return;
  }
  if (vec->length == 0) {
    AUBIO_WRN("aubio_func: empty vector");
    return;
  }
  for (i = 0; i < vec->length; i++) {
    AUBIO_ASSERT_BOUNDS(i, vec->length);
    vec->data[i] = ...;
  }
}
```

---

## Phase 6: Documentation & Guidelines (Priority: LOW)

### 6.1 Security Guidelines for Contributors

**Status:** ⏸️ PLANNED  
**Priority:** LOW  
**Effort:** 1 day  
**Files:** `CONTRIBUTING_SECURITY.md`

**Content to Create:**
- [ ] Array access best practices
- [ ] Required bounds checking patterns
- [ ] Memory allocation guidelines
- [ ] Input validation requirements
- [ ] String handling security
- [ ] Examples of common vulnerabilities
- [ ] Security checklist for PR reviews

### 6.2 Function Documentation Improvements

**Status:** ⏸️ PLANNED  
**Priority:** LOW  
**Effort:** 2-3 days  
**Files:** Various header files

**Functions Needing Better Docs:**
- [ ] `fvec_quadratic_peak_mag` - Document pos bounds requirement
- [ ] `fvec_peakpick` - Document caller must ensure `0 < pos < length-1`
- [ ] `interp_2` - Document wavetable padding requirement
- [ ] All spectral descriptor functions - Document input constraints
- [ ] All pitch detection functions - Document valid input ranges

**Documentation Format:**
```c
/**
 * Find peak magnitude using quadratic interpolation
 * 
 * @param x      Input vector
 * @param pos    Peak position (must be in range [1, x->length-2])
 * @return       Interpolated peak magnitude
 * 
 * @warning pos must allow access to pos-1 and pos+1
 * @pre     1 <= pos <= x->length - 2
 * @post    Returns 0 if preconditions not met
 * 
 * @note This function performs quadratic interpolation around the
 *       peak position to estimate the true peak magnitude.
 */
smpl_t fvec_quadratic_peak_mag (fvec_t *x, smpl_t pos);
```

### 6.3 Security Audit Documentation

**Status:** ⏸️ PLANNED  
**Priority:** LOW  
**Effort:** 4 hours  
**Files:** `SECURITY_AUDIT.md`

**Tasks:**
- [ ] Document security review process
- [ ] List tools and methods used
- [ ] Document all findings and resolutions
- [ ] Create security testing checklist
- [ ] Schedule regular security reviews

---

## Phase 7: CI/CD Integration (Priority: HIGH)

### 7.1 Sanitizer CI Jobs

**Status:** 🔄 IN PROGRESS  
**Priority:** HIGH  
**Effort:** 1 day  
**Files:** `.github/workflows/security.yml`

**Tasks:**
- [ ] Create dedicated security workflow
- [ ] Add ASAN build and test job
- [ ] Add UBSAN build and test job
- [ ] Add MSAN build and test job (if supported)
- [ ] Add Valgrind test job
- [ ] Make jobs required for PR merging

**Workflow Example:**
```yaml
name: Security Testing

on: [push, pull_request]

jobs:
  asan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        run: pip install meson ninja numpy
      - name: Configure
        run: meson setup builddir -Dtests=true -Db_sanitize=address
      - name: Build
        run: meson compile -C builddir
      - name: Test
        run: meson test -C builddir --print-errorlogs
        env:
          ASAN_OPTIONS: detect_leaks=1:symbolize=1
```

### 7.2 Static Analysis CI

**Status:** 🔄 IN PROGRESS  
**Priority:** MEDIUM  
**Effort:** 4 hours  
**Files:** `.github/workflows/static-analysis.yml`

**Tasks:**
- [ ] Add Clang-Tidy job
- [ ] Add Cppcheck job
- [ ] Add CodeQL job (already exists)
- [ ] Make findings fail CI on new issues
- [ ] Configure baseline for existing issues

### 7.3 Pre-commit Hooks

**Status:** ⏸️ PLANNED  
**Priority:** LOW  
**Effort:** 4 hours  
**Files:** `.pre-commit-config.yaml`

**Tasks:**
- [ ] Create pre-commit configuration
- [ ] Add Clang-Tidy hook
- [ ] Add Cppcheck hook
- [ ] Add format checker
- [ ] Document pre-commit setup
- [ ] Add to contributor guidelines

---

## Phase 8: Performance Validation (Priority: LOW)

### 8.1 Performance Benchmarking

**Status:** ⏸️ PLANNED  
**Priority:** LOW  
**Effort:** 1-2 days  
**Files:** `benchmarks/`

**Tasks:**
- [ ] Create benchmarking infrastructure
- [ ] Benchmark spectral analysis (with rolloff fix)
- [ ] Benchmark pitch detection (with schmitt fix)
- [ ] Benchmark overall audio processing pipeline
- [ ] Compare performance before/after security fixes
- [ ] Profile with `perf` or `gprof`
- [ ] Optimize any significant regressions

---

## Implementation Schedule

### Week 1: High Priority Security (Phase 1-2)
- **Days 1-2:** fvec_quadratic_peak_mag fix and verification
- **Days 3-4:** Build hardening and compiler flags
- **Day 5:** Debug build assertions

### Week 2: Memory Safety Testing (Phase 3)
- **Days 1-2:** ASAN integration and testing
- **Day 3:** Valgrind integration
- **Days 4-5:** UBSAN integration and fixes

### Week 3: Test Coverage (Phase 4)
- **Days 1-3:** Boundary condition tests
- **Days 4-5:** Memory leak detection and fixes

### Week 4: Static Analysis (Phase 5)
- **Days 1-3:** Clang-Tidy integration and fixes
- **Days 4-5:** Defensive programming patterns

### Week 5: CI/CD & Documentation (Phase 6-7)
- **Days 1-2:** Sanitizer CI jobs
- **Days 3-4:** Static analysis CI
- **Day 5:** Security documentation

### Week 6: Fuzz Testing & Performance (Phase 4.2, 8)
- **Days 1-3:** Fuzz testing infrastructure
- **Days 4-5:** Performance validation

---

## Success Metrics

### Security Metrics
- ✅ 0 critical vulnerabilities (achieved)
- ✅ 0 high vulnerabilities (achieved)
- 🎯 0 medium vulnerabilities (1 to verify: fvec_quadratic_peak_mag)
- 🎯 100% test coverage for boundary conditions
- 🎯 0 ASAN/UBSAN/Valgrind issues
- 🎯 0 static analysis warnings in new code

### Code Quality Metrics
- ✅ All tests passing (45/45)
- 🎯 All functions have proper bounds checking
- 🎯 All public APIs have input validation
- 🎯 All security-critical functions documented

### Process Metrics
- 🎯 Sanitizers running in CI
- 🎯 Static analysis running in CI
- 🎯 Security guidelines documented
- 🎯 Pre-commit hooks available

---

## Risk Management

### Identified Risks

1. **Performance Regression**
   - **Mitigation:** Benchmark before/after, optimize if needed
   - **Acceptance:** <5% performance impact acceptable for security

2. **False Positives in Static Analysis**
   - **Mitigation:** Configure suppressions carefully
   - **Acceptance:** Document and review each suppression

3. **Platform-Specific Issues**
   - **Mitigation:** Test on all supported platforms
   - **Acceptance:** Platform-specific fixes acceptable

4. **Test Maintenance Burden**
   - **Mitigation:** Focus on high-value tests
   - **Acceptance:** Tests must provide clear value

---

## Review & Maintenance

### Quarterly Review (Every 3 Months)
- Review open items
- Prioritize based on project needs
- Update implementation plan
- Run comprehensive security scan

### After Major Features
- Security review of new audio processing code
- Update test suite for new functionality
- Review for new vulnerability patterns

### Annual Comprehensive Audit
- External security tools review
- Full fuzzing campaign (48+ hours)
- Performance profiling
- Documentation review

---

## Tracking

| Phase | Priority | Status | Completion | Target Date |
|-------|----------|--------|------------|-------------|
| 1. Immediate Security | HIGH | 🔄 In Progress | 0% | Week 1 |
| 2. Build Hardening | HIGH | 🔄 In Progress | 0% | Week 1 |
| 3. Memory Safety Testing | HIGH | 🔄 In Progress | 0% | Week 2 |
| 4. Test Coverage | MEDIUM | 🔄 In Progress | 0% | Week 3 |
| 5. Static Analysis | MEDIUM | 🔄 In Progress | 0% | Week 4 |
| 6. Documentation | LOW | ⏸️ Planned | 0% | Week 5 |
| 7. CI/CD Integration | HIGH | 🔄 In Progress | 0% | Week 5 |
| 8. Performance | LOW | ⏸️ Planned | 0% | Week 6 |

---

## Appendix A: Security Fixes Already Applied

These items are documented in SECURITY_REVIEW.md as completed:

1. ✅ **Buffer over-read in aubio_sampler_load** (HIGH) - Fixed memory allocation and null termination
2. ✅ **Inconsistent null termination in new_aubio_tempo** (MEDIUM) - Fixed defensive programming
3. ✅ **Spectral rolloff out-of-bounds** (HIGH) - Fixed off-by-one error  
4. ✅ **Pitch Schmitt trigger out-of-bounds** (HIGH) - Fixed loop bounds

All fixes validated with test suite (45/45 tests passing).

---

## Appendix B: Tools & Technologies

### Build System
- **Meson** - Primary build system
- **Ninja** - Build backend
- **vcpkg** - Dependency management

### Security Testing
- **AddressSanitizer (ASAN)** - Memory error detection
- **UndefinedBehaviorSanitizer (UBSAN)** - Undefined behavior detection
- **MemorySanitizer (MSAN)** - Uninitialized memory detection
- **Valgrind** - Memory leak and error detection
- **libFuzzer/AFL++** - Fuzz testing

### Static Analysis
- **Clang-Tidy** - C/C++ linter and static analyzer
- **Cppcheck** - Static analysis tool
- **CodeQL** - Semantic code analysis (already integrated)
- **Coverity Scan** - Commercial-grade static analysis (free for OSS)
- **Infer** - Facebook's static analyzer

### CI/CD
- **GitHub Actions** - Automation platform
- **pre-commit** - Git hook framework

---

**Last Updated:** 2025-11-14  
**Next Review:** 2025-11-21 (Weekly during implementation)
