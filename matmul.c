#include <stdio.h>
#include <stdlib.h>

#define SIZE 10

void matrixMultiplication(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
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
    matrixMultiplication(A, B, C);

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
