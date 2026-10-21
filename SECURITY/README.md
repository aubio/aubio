# Security Documentation

This directory contains comprehensive security documentation for the aubio-ledfx project. These documents describe security measures, best practices, and guidelines for maintaining memory safety and preventing vulnerabilities in the C codebase.

## Quick Reference

### 🎯 Strategic Planning & Leadership
- **[COMPREHENSIVE_SECURITY_META_REVIEW.md](COMPREHENSIVE_SECURITY_META_REVIEW.md)** - Strategic security analysis and roadmap
  - Use when: Understanding overall security journey and strategy
  - Use when: Planning security investments and priorities
  - Use when: Reporting to executive leadership
  - Use when: Annual or quarterly security reviews
  - **Audience**: Principal Engineers, Security Teams, Leadership

### 🔍 When Reviewing Code
- **[REVIEW.md](REVIEW.md)** - Comprehensive security review findings and vulnerability fixes
  - Use when: Understanding past security issues and fixes
  - Use when: Conducting security audits
  - Use when: Reviewing similar code patterns

### 🛡️ When Writing C Code
- **[DEFENSIVE_PROGRAMMING.md](DEFENSIVE_PROGRAMMING.md)** - Defensive programming patterns and assertion macros
  - Use when: Writing new C functions
  - Use when: Adding memory safety checks
  - Use when: Validating input parameters
  - **Required**: All new C code must use defensive programming patterns

### 🔨 When Building
- **[HARDENING.md](HARDENING.md)** - Compiler security flags and build hardening
  - Use when: Configuring builds
  - Use when: Setting up development environments
  - Use when: Understanding compiler protections

### 🧪 When Testing
- **[SANITIZERS.md](SANITIZERS.md)** - Runtime sanitizers for detecting memory issues
  - Use when: Running tests
  - Use when: Debugging memory issues
  - Use when: CI/CD setup
  - **Required**: All code changes must pass sanitizer tests

### 📊 Security Status
- **[STRENGTHENING_SUMMARY.md](STRENGTHENING_SUMMARY.md)** - Overall security improvements and metrics
  - Use when: Understanding current security posture
  - Use when: Tracking security progress
  - Use when: Planning security work

## Document Overview

### COMPREHENSIVE_SECURITY_META_REVIEW.md
**Purpose**: Strategic security meta-analysis synthesizing all historical security work

**Contains**:
- Complete security timeline (2018-2025)
- Vulnerability pattern analysis (5 vulnerabilities, CWE classification)
- Security work evolution (4 phases: reactive → comprehensive → proactive → documentation)
- Current codebase assessment (22K lines, 124 files, 75% reviewed)
- Root cause analysis (why vulnerabilities existed, systemic issues)
- Defense-in-depth layers (4-layer security model)
- Patterns of success and failure
- Strategic remediation roadmap (10 phases)
- Comprehensive recommendations for all stakeholders
- Security metrics and KPIs

**Size**: 929 lines | **Classification**: Strategic Planning Document

**When to use**:
- Understanding the complete security history and context
- Planning security roadmap and resource allocation
- Preparing for security audits or compliance reviews
- Reporting to leadership or stakeholders
- Learning from past security work patterns
- Setting security priorities and goals

### REVIEW.md
**Purpose**: Historical record of security vulnerabilities found and fixed

**Contains**:
- Issue #433: NULL pointer dereference fixes (14 files)
- Issue #421: Buffer over-read fixes (2 files)
- PR #318: Spectral rolloff out-of-bounds fix
- Comprehensive codebase review results
- Testing and validation results

**Size**: 846 lines | **Severity**: Contains HIGH severity fixes

**When to use**:
- Before modifying similar code patterns
- When investigating security-related bugs
- During security audits
- When reviewing memory allocation code

### DEFENSIVE_PROGRAMMING.md
**Purpose**: Guide for writing secure, defensive C code

**Contains**:
- Security assertion macros (AUBIO_ASSERT_*)
- Input validation patterns
- Array access best practices
- Loop bounds checking
- Null pointer handling

**Size**: 445 lines | **Audience**: All C developers

**When to use**:
- **REQUIRED** when writing new C functions
- When adding bounds checking
- When validating user input
- When working with pointers and arrays
- When implementing memory safety checks

**Critical for**:
- All vector operations (fvec, cvec, fmat, lvec)
- All object constructors (new_aubio_*)
- All array indexing operations
- All pointer dereferences

### HARDENING.md
**Purpose**: Build system security configuration

**Contains**:
- Stack protection (-fstack-protector-strong)
- Fortify source (-D_FORTIFY_SOURCE=2)
- Format string security
- Position Independent Executables (PIE)
- Build configuration examples

**Size**: 280 lines | **Audience**: Build engineers, DevOps

**When to use**:
- Setting up development environments
- Configuring CI/CD pipelines
- Troubleshooting build issues
- Understanding compiler protections
- Platform-specific builds

### SANITIZERS.md
**Purpose**: Runtime memory safety testing

**Contains**:
- AddressSanitizer (ASAN) - buffer overflows, use-after-free
- UndefinedBehaviorSanitizer (UBSAN) - undefined behavior
- MemorySanitizer (MSAN) - uninitialized memory
- ThreadSanitizer (TSAN) - data races
- LeakSanitizer (LSAN) - memory leaks

**Size**: 462 lines | **Audience**: All developers, CI/CD

**When to use**:
- **REQUIRED** before committing C code changes
- When debugging memory issues
- When tests fail mysteriously
- During code review
- In CI/CD pipelines

**Performance Impact**: 2x slowdown (ASAN), 1.2x (UBSAN)

### STRENGTHENING_SUMMARY.md
**Purpose**: Overall security improvement summary and metrics

**Contains**:
- Vulnerability fixes summary
- Security hardening implementation
- Testing infrastructure
- Documentation overview
- Metrics and statistics
- Future work roadmap

**Size**: 448 lines | **Audience**: Project managers, security teams

**When to use**:
- Understanding overall security posture
- Reviewing security progress
- Planning security work
- Quarterly security reviews
- Stakeholder reporting

## Quick Start Guides

### For New C Code

```c
#include "aubio_priv.h"

void my_new_function(fvec_t *vec, uint_t index) {
  // 1. Validate inputs
  AUBIO_ASSERT_NOT_NULL(vec);
  AUBIO_ASSERT_BOUNDS(index, vec->length);

  // 2. Safe to proceed
  vec->data[index] = 0.0;
}
```

**See**: [DEFENSIVE_PROGRAMMING.md](DEFENSIVE_PROGRAMMING.md)

### For Building with Security

```bash
# Standard secure build
meson setup builddir -Dsecurity_hardening=true -Dtests=true
meson compile -C builddir

# With sanitizers (recommended)
meson setup builddir -Db_sanitize=address,undefined -Dtests=true
meson compile -C builddir
meson test -C builddir
```

**See**: [HARDENING.md](HARDENING.md), [SANITIZERS.md](SANITIZERS.md)

### For Code Review

**Checklist**:
- [ ] All array accesses have bounds checks
- [ ] All allocations checked for NULL
- [ ] All loops have correct bounds
- [ ] Defensive assertions added (debug mode)
- [ ] Tests pass with ASAN + UBSAN
- [ ] Similar patterns reviewed in REVIEW.md

**See**: [REVIEW.md](REVIEW.md), [DEFENSIVE_PROGRAMMING.md](DEFENSIVE_PROGRAMMING.md)

## Security Workflow

```
Write Code → Add Assertions → Build → Test with Sanitizers → Review → Commit
    ↓             ↓            ↓              ↓              ↓        ↓
  Follow      DEFENSIVE_   HARDENING    SANITIZERS      REVIEW    CI/CD
  patterns    PROGRAMMING     .md           .md           .md    validates
```

## Memory Safety Requirements

### Critical Functions Requiring Extra Care

**Vector Operations** (src/fvec.c, src/cvec.c, src/fmat.c, src/lvec.c):
- All functions must validate indices
- All functions must check for NULL
- See DEFENSIVE_PROGRAMMING.md for patterns

**Object Constructors** (src/**/new_aubio_*.c):
- Check all allocations for NULL
- Use "goto beach" cleanup pattern
- Validate parameters before allocating
- See REVIEW.md for examples

**Array Indexing** (anywhere accessing data[i]):
- Ensure i < length
- For lookahead (data[i+1]), ensure i < length-1
- Use AUBIO_ASSERT_BOUNDS in debug mode
- See DEFENSIVE_PROGRAMMING.md for patterns

**String Handling**:
- No strcpy, sprintf, gets, strcat
- Use strncpy with explicit null termination
- Allocate strnlen(...) + 1 bytes
- See REVIEW.md issue #421 for examples

## Testing Requirements

### Before Every Commit

```bash
# 1. Build with sanitizers
meson setup builddir -Db_sanitize=address,undefined -Dtests=true

# 2. Run all tests
meson test -C builddir --print-errorlogs

# 3. Verify no sanitizer errors
echo "Check output for ASAN/UBSAN errors"
```

**Required**: All tests must pass with zero sanitizer errors.

### Continuous Integration

All PRs automatically run:
- ASAN + UBSAN tests
- CodeQL security analysis
- Build on all platforms

**See**: [SANITIZERS.md](SANITIZERS.md) for CI/CD setup

## Vulnerability Reporting

### Fixed Vulnerabilities

- **Issue #433**: NULL pointer dereferences (HIGH) - ✅ Fixed
- **Issue #421**: Buffer over-reads (HIGH) - ✅ Fixed
- **PR #318**: Spectral rolloff OOB (HIGH) - ✅ Fixed
- **fvec_quadratic_peak_mag**: Bounds check (LOW) - ✅ Fixed

**Total**: 4 vulnerabilities fixed, 0 open

**See**: [REVIEW.md](REVIEW.md) for detailed analysis

### Reporting New Issues

For security vulnerabilities:
1. Do NOT open public GitHub issues
2. Use GitHub Security Advisories
3. Or contact maintainers privately

## Metrics

### Current Security Posture

| Metric | Status |
|--------|--------|
| CodeQL Alerts | ✅ 0 |
| Sanitizer Tests | ✅ 45/45 passing |
| Compiler Hardening | ✅ Enabled |
| Security Assertions | ✅ Implemented |
| Test Coverage | ✅ 100% pass rate |

### Documentation Coverage

| Document | Lines | Focus Area |
|----------|-------|------------|
| COMPREHENSIVE_SECURITY_META_REVIEW.md | 929 | Strategic analysis |
| REVIEW.md | 878 | Historical fixes |
| STRENGTHENING_SUMMARY.md | 478 | Status overview |
| DEFENSIVE_PROGRAMMING.md | 479 | Code patterns |
| SANITIZERS.md | 500 | Testing |
| HARDENING.md | 310 | Build config |
| README.md (this file) | 349 | Navigation |
| **Total** | **~3,900+** | **Comprehensive** |

## Contributing

### When Adding New Code

1. **Read**: [DEFENSIVE_PROGRAMMING.md](DEFENSIVE_PROGRAMMING.md)
2. **Add**: Assertion macros to all functions
3. **Test**: Run with sanitizers (see [SANITIZERS.md](SANITIZERS.md))
4. **Review**: Check [REVIEW.md](REVIEW.md) for similar patterns
5. **Build**: Use secure flags (see [HARDENING.md](HARDENING.md))

### When Reviewing Code

1. **Check**: All patterns in [DEFENSIVE_PROGRAMMING.md](DEFENSIVE_PROGRAMMING.md)
2. **Verify**: Sanitizer tests pass (see [SANITIZERS.md](SANITIZERS.md))
3. **Compare**: Against fixes in [REVIEW.md](REVIEW.md)
4. **Validate**: Build with security hardening (see [HARDENING.md](HARDENING.md))

## Related Documentation

- **Main README**: `/README.md` - Project overview
- **Build System**: `/doc/meson_reference.rst` - Build system details
- **vcpkg Integration**: `/doc/vcpkg_integration.md` - Dependency management
- **Copilot Instructions**: `/.github/copilot-instructions.md` - AI coding guidelines

## Maintenance

### Regular Tasks

**Weekly**:
- Monitor sanitizer CI results
- Review security updates

**Monthly**:
- Update security documentation
- Review new compiler features

**Quarterly**:
- Full security audit
- Update dependencies
- Review assertion coverage

### Last Updated

- **Date**: 2025-11-16
- **Version**: 0.5.0-alpha
- **Next Review**: 2025-11-23
- **Major Addition**: Comprehensive Security Meta-Review (929 lines)

## License

Same as aubio-ledfx main project (GPL-3.0).

---

**Note**: This directory contains critical security documentation. All developers working on the C codebase MUST be familiar with these documents, especially DEFENSIVE_PROGRAMMING.md and SANITIZERS.md.
