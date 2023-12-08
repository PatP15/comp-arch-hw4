#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdint.h>

#define SIZE 10

void serialMatrixMult(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void vecMatrixMult(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            __m256d sum = _mm256_setzero_pd(); // Initialize sum vector to zero

            for (int k = 0; k < SIZE; k += 4) {
                // Load 4 double-precision elements from A and B
                __m256d a = _mm256_loadu_pd(&A[i][k]);
                __m256d b = _mm256_loadu_pd(&B[k][j]);

                // Perform element-wise multiplication and add to sum
                sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
            }

            // Horizontally add the elements of the sum vector and store the result in C[i][j]
            sum = _mm256_hadd_pd(sum, sum);
            __m128d sum_high = _mm256_extractf128_pd(sum, 1);
            __m128d result = _mm_add_pd(_mm256_castpd256_pd128(sum), sum_high);
            C[i][j] = _mm_cvtsd_f64(result);
        }
    }
}

static inline uint64_t rdtsc(void) {

    unsigned int lo, hi;
    asm volatile("mfence");
    asm volatile("rdtsc": "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

int main() {
    // Allocate memory for matrices A, B, and C
    double (*A)[SIZE] = malloc(SIZE * sizeof(*A));
    double (*B)[SIZE] = malloc(SIZE * sizeof(*B));
    double (*C)[SIZE] = malloc(SIZE * sizeof(*C));

    // Initialize matrices A and B with random values
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            //A[i][j] = (double)rand() / RAND_MAX;
            //B[i][j] = (double)rand() / RAND_MAX;
	    A[i][j] = i;
	    B[i][j] = j;
        }
    }

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%lf ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%lf ", B[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Perform matrix multiplication
    serialMatrixMult(A, B, C);

    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%lf ", C[i][j]);
        }
        printf("\n\n");
    }

    vecMatrixMult(A, B, C);
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%lf ", C[i][j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);

    return 0;
}
