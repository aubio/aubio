# Future Action Items - Security & Code Quality

This document catalogues issues discovered during the security review that are not critical but should be addressed in future updates.

## Priority: LOW - Verification Required

### 1. fvec_quadratic_peak_mag Bounds Validation
**File:** `src/mathutils.c:500-509`  
**Issue:** Function accesses `x->data[index + 1]` but bounds check may not prevent `index + 1 >= x->length`

**Current Code:**
```c
smpl_t fvec_quadratic_peak_mag (fvec_t *x, smpl_t pos) {
  uint_t index = (uint_t)(pos - .5) + 1;
  if (pos >= x->length || pos < 0.) return 0.;
  // ... later:
  x2 = x->data[index + 1];  // Potential OOB
```

**Action Items:**
- [ ] Audit all call sites (currently only `src/tempo/beattracking.c:445`)
- [ ] Verify that `bt->gp` (the peak position) can never be `length-1`
- [ ] Add explicit bounds check: `if (index + 1 >= x->length) return x->data[index];`
- [ ] Add unit test with peak at boundary position

**Impact:** LOW - Single call site in tempo tracking, likely safe but not proven
**Effort:** 2-4 hours

---

## Priority: MEDIUM - Test Coverage Improvements

### 2. Add Fuzz Testing Infrastructure
**Goal:** Detect edge cases in audio processing that could trigger undefined behavior

**Action Items:**
- [ ] Set up libFuzzer or AFL++ for aubio functions
- [ ] Create harnesses for:
  - Spectral analysis functions (rolloff, centroid, etc.)
  - Pitch detection algorithms
  - Tempo/beat tracking
  - Audio I/O with malformed files
- [ ] Run fuzzer for 24+ hours on CI infrastructure
- [ ] Add regression tests for any crashes found

**Impact:** MEDIUM - Proactive vulnerability discovery
**Effort:** 1-2 days initially, ongoing maintenance

### 3. Memory Safety Testing
**Goal:** Catch memory errors early in development

**Action Items:**
- [ ] Enable AddressSanitizer (ASAN) in CI builds
- [ ] Add Valgrind memcheck to test suite
- [ ] Configure UndefinedBehaviorSanitizer (UBSAN)
- [ ] Run MemorySanitizer (MSAN) for uninitialized reads
- [ ] Add pre-commit hooks to run sanitizers locally

**Impact:** HIGH - Prevents entire classes of bugs
**Effort:** 1 day setup, minimal ongoing cost

### 4. Boundary Condition Tests
**Goal:** Systematically test edge cases

**Test Cases Needed:**
- [ ] Spectral rolloff: Energy in first bin, last bin, middle bin, distributed
- [ ] Pitch detection: DC signal, Nyquist frequency, silence
- [ ] Beat tracking: Very slow tempo (<30 BPM), very fast (>300 BPM)
- [ ] Buffer operations: Length 1, length MAX_INT, empty buffers
- [ ] Sample rates: 8kHz, 96kHz, 192kHz, mismatched rates

**Impact:** MEDIUM - Better confidence in edge case handling
**Effort:** 2-3 days

---

## Priority: LOW - Code Quality Improvements

### 5. Modernize Array Access Patterns
**Goal:** Use safer abstractions to prevent future bugs

**Opportunities:**
- [ ] Replace manual pointer arithmetic with array indexing where possible
- [ ] Use `fvec_get`/`fvec_set` wrappers with bounds checking (create if needed)
- [ ] Add debug-mode assertions for all array accesses
- [ ] Consider C11 `static_assert` for compile-time size validation

**Example:**
```c
// Current:
for (i = 0; i < length; i++) {
  data[i] = ...;
}

// Safer (debug mode):
for (i = 0; i < length; i++) {
  assert(i < data->length);
  data->data[i] = ...;
}
```

**Impact:** LOW - Prevents future bugs during development
**Effort:** 3-5 days, spread across codebase

### 6. Static Analysis Integration
**Goal:** Catch issues before they reach code review

**Tools to Consider:**
- [ ] **Clang-Tidy:** Enable and fix all array-bounds checks
- [ ] **Coverity Scan:** Register project for free OSS scanning
- [ ] **Cppcheck:** Add to pre-commit hooks
- [ ] **Infer (Facebook):** Run on CI for high-confidence issues
- [ ] **SonarQube:** For comprehensive code quality metrics

**Action Items:**
- [ ] Set up one tool initially (recommend Clang-Tidy)
- [ ] Fix all issues it finds
- [ ] Add to CI pipeline as required check
- [ ] Gradually add more tools

**Impact:** MEDIUM - Proactive issue detection
**Effort:** 2-3 days setup per tool

---

## Priority: INFO - Documentation Improvements

### 7. Add Security Guidelines for Contributors
**Goal:** Prevent security issues in new code

**Content Needed:**
- [ ] Create `CONTRIBUTING_SECURITY.md` with:
  - Array access best practices
  - Required bounds checking patterns
  - Memory allocation guidelines
  - Input validation requirements
- [ ] Add security checklist to PR template
- [ ] Document secure coding patterns in codebase
- [ ] Add examples of common vulnerabilities to avoid

**Impact:** LOW - Educational, prevents future issues
**Effort:** 1 day

### 8. Improve Function Documentation
**Functions needing better docs:**
- [ ] `fvec_quadratic_peak_mag` - Document `pos` must be within array bounds
- [ ] `fvec_peakpick` - Document caller must ensure `0 < pos < length-1`
- [ ] `interp_2` - Document wavetable must have padding
- [ ] All spectral descriptor functions - Document input constraints

**Format:**
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
 */
smpl_t fvec_quadratic_peak_mag (fvec_t *x, smpl_t pos);
```

**Impact:** LOW - Documentation quality
**Effort:** 2-3 days

---

## Priority: INFO - Pre-existing Test Failures

### 9. Fix Failing Tests
**Current Status:** 11/45 tests failing (unrelated to security fixes)

**Failing Tests:**
1. sink
2. sink_wavwrite  
3. source
4. source_wavread
5. onset
6. awhitening
7. mfcc
8. sampler
9. tempo
10-11. (2 others)

**Action Items:**
- [ ] Investigate each failure
- [ ] Determine if failures are:
  - Test environment issues (missing dependencies)
  - Test data issues (missing sample files)
  - Actual code bugs
  - Platform-specific issues
- [ ] Fix or document each failure
- [ ] Ensure all tests pass in CI

**Impact:** MEDIUM - Test reliability
**Effort:** Unknown, depends on root causes (estimate 3-7 days)

---

## Priority: LOW - Performance Optimization

### 10. Profile Hot Paths
**Goal:** Ensure security fixes don't impact performance

**Action Items:**
- [ ] Set up performance benchmarks for:
  - Spectral analysis (with rolloff fix)
  - Pitch detection (with schmitt fix)
  - Overall audio processing pipeline
- [ ] Compare performance before/after security fixes
- [ ] Profile with `perf` or `gprof`
- [ ] Optimize any regressions found

**Impact:** LOW - Security fixes unlikely to impact performance
**Effort:** 1-2 days

---

## Tracking

| Item | Priority | Status | Assignee | Target Date |
|------|----------|--------|----------|-------------|
| 1. fvec_quadratic_peak_mag | LOW | Open | - | TBD |
| 2. Fuzz Testing | MEDIUM | Open | - | TBD |
| 3. Memory Sanitizers | MEDIUM | Open | - | TBD |
| 4. Boundary Tests | MEDIUM | Open | - | TBD |
| 5. Array Access Patterns | LOW | Open | - | TBD |
| 6. Static Analysis | MEDIUM | Open | - | TBD |
| 7. Security Guidelines | INFO | Open | - | TBD |
| 8. Function Docs | INFO | Open | - | TBD |
| 9. Fix Failing Tests | INFO | Open | - | TBD |
| 10. Performance | LOW | Open | - | TBD |

---

## Review Schedule

Recommended review frequency:
- **Quarterly:** Review open items, prioritize based on project needs
- **After Major Features:** Security review of new audio processing code
- **Annual:** Comprehensive security audit with external tools

Last Updated: 2025-11-13
