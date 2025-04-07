#include "rdtsc.h"

#include <argp.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Args {
  size_t length;
  size_t nreps;
  bool block;
  size_t unroll_factor;
};

static struct argp_option options[] = {
  {"length", 'n', "size_t", 0, "Length of each vector"},
  {"nreps", 'r', "size_t", 0, "Number of repetitions"},
  {"block", 'b', NULL, 0, "Compute block dot products (versus a single dot product)"},
  {"unroll_factor", 'm', "size_t", 0, "Runtime unrolling factor for dot product"},
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
  struct Args *args = state->input;
  switch (key) {
  case ARGP_KEY_INIT:
    args->length = 100;
    args->nreps = 10;
    args->block = false;
    args->unroll_factor = 4;
    break;
  case 'n':
    args->length = strtol(arg, NULL, 10);
    break;
  case 'r':
    args->nreps = strtol(arg, NULL, 10);
    break;
  case 'b':
    args->block = true;
    break;
  case 'm':
    args->unroll_factor = strtol(arg, NULL, 10);
    break;
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

// This part is to answer Part 1
#define COMP_TIME_M 4

// dot_opt_comp_time_m uses a compile-time unrolling factor (Part 1, Q4)
double dot_opt_comp_time_m(size_t n, const double *a, const double *b) {

  // check if n is divided evenly by the compile-time unrolling factor
  if (n % COMP_TIME_M != 0) {
    printf("Error, the compile time unrolling factor must be perfectly contained in n \n");
    return -1;
  }

  double sum = 0, sums[n / COMP_TIME_M];
  memset(sums, 0, sizeof(sums));

  for (size_t i=0; i<n; i+=COMP_TIME_M) {
    for (size_t j=0; j<COMP_TIME_M; j++) {
      sums[j] += a[i+j] * b[i+j];
    }
  }
  for (size_t j=0; j < n/COMP_TIME_M; j++) {
    sum += sums[j];
  }
  return sum;
}

// dot_opt_run_time_m uses a runtime unrolling factor (Part 1, Q5)
double dot_opt_run_time_m(size_t n, const double *a, const double *b, const size_t m) {

  // check if n is divided evenly by the compile-time unrolling factor
  if (n % m != 0) {
    printf("Error, the compile time unrolling factor must be perfectly contained in n \n");
    return -1;
  }

  double sum = 0, sums[n / m];

  // initialize the dynamically created sums array to zero
  memset(sums, 0, sizeof(sums));
  // printf("Before for-loop \n");
  for (size_t i=0; i<n; i+=m) {
    // printf("Inside first for-loop \n");
    for (size_t j=0; j < m; j++) {
      // printf("Inside 2nd for-loop. I is %d, j is %d \n", i, j);
      sums[j] += a[i+j] * b[i+j];
    }
  }
  for (size_t j=0; j < n/m; j++) {
    // printf("Inside the reduction for-loop. j is %d \n", j);
    sum += sums[j];
  }
  return sum;
}
// This part ends to answer Part 1

__attribute((noinline))
static double dot_ref(size_t n, const double *a, const double *b) {
  double sum = 0;
  for (size_t i=0; i<n; i++)
    sum += a[i] * b[i];
  return sum;
}

// The following even-odd function given in the assignment tries to alleviate some data dependency from one loop iteration to another.
// Note: the following function is NOT correct for all values of n. In fact, for n odd we incur in out-of-bound access when we perform the second partial sum sum1 += a[i+1] * b[i+1]
double dot_opt_even_odd(size_t n, const double *a, const double *b) {
  double sum0 = 0, sum1 = 0;
  for (size_t i=0; i<n; i+=2) {
    sum0 += a[i+0] * b[i+0];
    sum1 += a[i+1] * b[i+1];
  }
  return sum0 + sum1;
}

__attribute((noinline))
static double dot_opt1(size_t n, const double *a, const double *b) {
  double sums[4] = {};
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    for (size_t i=id; i<n; i+=4)
      sums[id] += a[i] * b[i];
  }
  for (size_t j=1; j<4; j++) sums[0] += sums[j];
  return sums[0];
}

// dot_opt1_simd is the dot_opt1 version to which we added the #pragma omp simd directive
__attribute((noinline))
static double dot_opt1_simd(size_t n, const double *a, const double *b) {
  double sums[4] = {};
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    #pragma omp simd
    for (size_t i=id; i<n; i+=4)
      sums[id] += a[i] * b[i];
  }
  #pragma omp simd
  for (size_t j=1; j<4; j++) sums[0] += sums[j];
  return sums[0];
}

__attribute((noinline))
static double dot_opt2(size_t n, const double *a, const double *b) {
  double sums[4] = {};
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    #pragma omp for
    for (size_t i=0; i<n; i++)
      sums[id] += a[i] * b[i];
  }
  for (size_t j=1; j<4; j++) sums[0] += sums[j];
  return sums[0];
}

// dot_opt2_simd is the dot_opt2 version to which we added the #pragma omp simd directive
__attribute((noinline))
static double dot_opt2_simd(size_t n, const double *a, const double *b) {
  double sums[4] = {};
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    int id = omp_get_thread_num();
    #pragma omp for simd
    for (size_t i=0; i<n; i++)
      sums[id] += a[i] * b[i];
  }
  #pragma omp simd
  for (size_t j=1; j<4; j++) sums[0] += sums[j];
  return sums[0];
}

__attribute((noinline))
static double dot_opt3(size_t n, const double *a, const double *b) {
  double sum = 0;
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    #pragma omp for reduction(+:sum)
    for (size_t i=0; i<n; i++)
      sum += a[i] * b[i];
  }
  return sum;
}

// dot_opt3_simd is the dot_opt3 version to which we added the #pragma omp simd directive
__attribute((noinline))
static double dot_opt3_simd(size_t n, const double *a, const double *b) {
  double sum = 0;
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    #pragma omp for simd reduction(+:sum)
    for (size_t i=0; i<n; i++)
      sum += a[i] * b[i];
  }
  return sum;
}

static void report_dot(const char *name, ticks_t start_ticks, size_t flops, double result) {
  ticks_t ticks = rdtsc() - start_ticks;
  double rate = 1.*flops / ticks;
  if (fabs(result - flops) > 1e-10)
    printf("Result %f failed to validate with expected value %ld\n", result, flops);
  printf("%8s\t%ld\t%lld\t%8.2f\t\n", name, flops, ticks, rate);
}

#define REPORT_DOT(f) do {                                              \
    for (int rep=0; rep<args.nreps; rep++) {                            \
      ticks_t ticks_start = rdtsc();                                    \
      report_dot(#f, ticks_start, 2*args.length, f(args.length, a, b)); \
    }                                                                   \
  } while (0)

// The following macro is to test the perfomance of the runtime loop unrolling constant to answer Part 1.5
#define REPORT_DOT_RUNTIME(f, m) do {                                              \
  for (int rep=0; rep<args.nreps; rep++) {                            \
    ticks_t ticks_start = rdtsc();                                    \
    report_dot(#f, ticks_start, 2*args.length, f(args.length, a, b, m)); \
  }                                                                   \
} while (0)

// Dimensions of the matrices for block dot products.
#define J 8
#define K 4

// Performs the operation
//   C = A * B
// where A and B have shape (J,n) and (n,K) respectively.
// This reference version stores A as row-major and B as column-major.
static void bdot_ref(size_t n, const double *a, const double *b, double *c) {
  for (size_t j=0; j<J; j++) {
    for (size_t k=0; k<K; k++) {
      c[j*K+k] = dot_ref(n, &a[j*n], &b[k*n]);
    }
  }
}

// bdot_opt1 internally calls dot_opt1
static void bdot_opt1(size_t n, const double *a, const double *b, double *c) {

  for (size_t j=0; j<J; j++) {
    for (size_t k=0; k<K; k++) {
      c[j*K+k] = dot_opt1(n, &a[j*n], &b[k*n]);
    }
  }
}

// bdot_opt1_simd internally calls dot_opt1_simd
static void bdot_opt1_simd(size_t n, const double *a, const double *b, double *c) {

  for (size_t j=0; j<J; j++) {
    for (size_t k=0; k<K; k++) {
      c[j*K+k] = dot_opt1_simd(n, &a[j*n], &b[k*n]);
    }
  }
}

// bdot_opt2 internally calls dot_opt2
static void bdot_opt2(size_t n, const double *a, const double *b, double *c) {

  for (size_t j=0; j<J; j++) {
    for (size_t k=0; k<K; k++) {
      c[j*K+k] = dot_opt2(n, &a[j*n], &b[k*n]);
    }
  }
}

// bdot_opt2_simd internally calls dot_opt2_simd
static void bdot_opt2_simd(size_t n, const double *a, const double *b, double *c) {

  for (size_t j=0; j<J; j++) {
    for (size_t k=0; k<K; k++) {
      c[j*K+k] = dot_opt2_simd(n, &a[j*n], &b[k*n]);
    }
  }
}

// bdot_opt3 internally calls dot_opt3
static void bdot_opt3(size_t n, const double *a, const double *b, double *c) {

  for (size_t j=0; j<J; j++) {
    for (size_t k=0; k<K; k++) {
      c[j*K+k] = dot_opt3(n, &a[j*n], &b[k*n]);
    }
  }
}

// bdot_opt3_simd internally calls dot_opt3_simd
static void bdot_opt3_simd(size_t n, const double *a, const double *b, double *c) {

  for (size_t j=0; j<J; j++) {
    for (size_t k=0; k<K; k++) {
      c[j*K+k] = dot_opt3_simd(n, &a[j*n], &b[k*n]);
    }
  }
}

// bdot_opt3_simd_reordered performs the J * K (=32) pairwise products swapping the k,j indices
static void bdot_opt3_simd_reordered(size_t n, const double *a, const double *b, double *c) {

  for (size_t k=0; k<K; k++) {
    for (size_t j=0; j<J; j++) {
      c[j*K+k] = dot_opt3_simd(n, &a[j*n], &b[k*n]);
    }
  }
}


static void init_bdot(size_t n, double *a, size_t ajstride, size_t aistride,
                      double *b, size_t bistride, size_t bkstride) {
  for (size_t i=0; i<n; i++) {
    for (size_t j=0; j<J; j++)
      a[i*aistride + j*ajstride] = 1000*(i+1) + j+1;
    for (size_t k=0; k<K; k++)
      b[i*bistride + k*bkstride] = 1./(1000*(i+1) + k+1);
  }
}

static void report_bdot(const char *name, ticks_t start_ticks, size_t flops,
                        const double *result, int jstride, int kstride,
                        const double *ref_result) {
  ticks_t ticks = rdtsc() - start_ticks;
  double rate = 1.*flops / ticks;
  if (result && ref_result && result != ref_result) {
    for (int j=0; j<J; j++) {
      for (int k=0; k<K; k++) {
        if (fabs(result[j*jstride + k*kstride] - ref_result[j*K+k]) > 1e-10) {
          printf("Result[%d,%d] = %f failed to validate with expected value %f\n", j, k, result[j*jstride + k*kstride], ref_result[j*K+k]);
        }
      }
    }
  }
  printf("%s\t%ld\t%lld\t%8.2f\t\n", name, flops, ticks, rate);
}

#define REPORT_BDOT(f, c, jstride, kstride, c_ref) do {                 \
    for (int rep=0; rep<args.nreps; rep++) {                            \
      ticks_t ticks_start = rdtsc();                                    \
      f(args.length, a, b, c);                                          \
      report_bdot(#f, ticks_start, 2*J*K*args.length, c, jstride, kstride, c_ref); \
    }                                                                   \
  } while (0)

int main(int argc, char **argv) {
  struct Args args;
  struct argp argp = {options, parse_opt, NULL, NULL};
  argp_parse(&argp, argc, argv, 0, 0, &args);
  size_t n = args.length;

  switch (args.block) {
  case false: {
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    for (size_t i=0; i<n; i++) {
      a[i] = 2.*(i+1);
      b[i] = 1./(i+1);
    }

    printf("  Name  \tflops\tticks\tflops/tick\n");
    REPORT_DOT(dot_ref);
    REPORT_DOT(dot_opt1);
    REPORT_DOT(dot_opt2);
    REPORT_DOT(dot_opt3);
    // To answer Part 1, we will test the excution for the even-odd, and compile/runtime loop unrolling constant versions
    REPORT_DOT(dot_opt_even_odd);
    REPORT_DOT(dot_opt_comp_time_m);
    REPORT_DOT_RUNTIME(dot_opt_run_time_m, args.unroll_factor);

    free(a); free(b);
  } break;
  case true: {
    // Initialize the matrices (as flattened vectors)
    double *a = malloc(J * n * sizeof(double));
    double *b = malloc(K * n * sizeof(double));
    double *c = malloc(J * K * sizeof(double));
    double *c_ref = malloc(J * K * sizeof(double));

    printf("Name    \tflops\tticks\tflops/tick\n");
    init_bdot(args.length, a, n, 1, b, 1, n);
    REPORT_BDOT(bdot_ref, c_ref, K, 1, c_ref);

    // You may initialize a and b differently and call more variants by editing
    // the two lines below, or by creating new variants.
    init_bdot(args.length, a, n, 1, b, 1, n);
    // To answer Part 2, we will test the excution for the three different blocked opt versions
    REPORT_BDOT(bdot_opt1, c, K, 1, c_ref);
    REPORT_BDOT(bdot_opt2, c, K, 1, c_ref);
    REPORT_BDOT(bdot_opt3, c, K, 1, c_ref);
    REPORT_BDOT(bdot_opt1_simd, c, K, 1, c_ref);
    REPORT_BDOT(bdot_opt2_simd, c, K, 1, c_ref);
    REPORT_BDOT(bdot_opt3_simd, c, K, 1, c_ref);
    REPORT_BDOT(bdot_opt3_simd_reordered, c, K, 1, c_ref);

    free(a); free(b); free(c); free(c_ref);
  } break;
  }
  return 0;
}
