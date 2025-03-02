#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static void triad(int N, double *a, const double *b, double scalar, const double *c) {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (int i=0; i<N; i++) {
            a[i] = b[i] + scalar * c[i];
            printf("Index i: %d, Thread id: %d, a[i]: %f \n", i, id, a[i]);
        }
    }
}

void strided_triad(int N, double *a, const double *b, double scalar, const double *c) {
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        for (int i=id; i<N; i+=num_threads) {
            a[i] = b[i] + scalar * c[i];
            printf("Index i: %d, Thread id: %d, a[i]: %f \n", i, id, a[i]);
        }
    }
}

static void initialize_array(int N, double *a, const double s) {
    for (int i=0; i<N; i++)
        a[i] = s;
}


int main() {

    // Initialize vectors and scalars
    int N = 10;
    double *a = calloc(N, sizeof(double));
    double *b = calloc(N, sizeof(double));
    double *c = calloc(N, sizeof(double));

    double s = 2.0;

    initialize_array(N, b, 1.0);
    initialize_array(N, c, 2.0);

    printf("Call to triad:\n");
    triad(N, a, b, s, c);

    // printf("Call to strided_triad:\n");
    // strided_triad(N, a, b, s, c);

    free(a); free(b); free(c);

    return 0;
}
