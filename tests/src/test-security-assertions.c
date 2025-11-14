/* Test file for security assertions in debug builds */
#define AUBIO_UNSTABLE 1
#include "aubio.h"
#include <stdio.h>

/* This test verifies that security assertions work in debug builds.
 * It should only be run manually, as it intentionally triggers assertions
 * which will abort the program.
 */

int test_bounds_check(void) {
  fvec_t *vec = new_fvec(10);
  smpl_t val;
  
  fprintf(stdout, "Testing bounds check...\n");
  
  // This should work fine
  fvec_set_sample(vec, 5.0, 5);
  val = fvec_get_sample(vec, 5);
  fprintf(stdout, "Valid access: vec[5] = %f\n", val);
  
  // This should trigger AUBIO_ASSERT_BOUNDS in debug builds
  fprintf(stdout, "Attempting out-of-bounds access (should abort in debug)...\n");
  val = fvec_get_sample(vec, 10); // Out of bounds!
  fprintf(stdout, "ERROR: Should not reach here! val = %f\n", val);
  
  del_fvec(vec);
  return 1; // Fail - should have aborted
}

int test_null_check(void) {
  smpl_t val;
  
  fprintf(stdout, "Testing null pointer check...\n");
  
  // This should trigger AUBIO_ASSERT_NOT_NULL in debug builds
  fprintf(stdout, "Attempting null pointer access (should abort in debug)...\n");
  val = fvec_get_sample(NULL, 0);
  fprintf(stdout, "ERROR: Should not reach here! val = %f\n", val);
  
  return 1; // Fail - should have aborted
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stdout, "Usage: %s <test_name>\n", argv[0]);
    fprintf(stdout, "Tests:\n");
    fprintf(stdout, "  bounds  - Test bounds checking\n");
    fprintf(stdout, "  null    - Test null pointer checking\n");
    fprintf(stdout, "\nNote: These tests will abort in debug builds (expected behavior)\n");
    return 0;
  }
  
  const char *test = argv[1];
  
  if (strcmp(test, "bounds") == 0) {
    return test_bounds_check();
  } else if (strcmp(test, "null") == 0) {
    return test_null_check();
  } else {
    fprintf(stderr, "Unknown test: %s\n", test);
    return 1;
  }
}
