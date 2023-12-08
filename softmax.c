#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VECTOR_LENGTH 10

void softmax(double *x, int size) {
    double max_x = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_x) {
            max_x = x[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        x[i] = exp(x[i] - max_x);
        sum += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

int main() {
    double *x = malloc(VECTOR_LENGTH * sizeof(double));

    // Initialize vector x with random values
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        x[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        printf("%lf ", x[i]);
    }
    printf("\n");

    // Softmax computation
    softmax(x, VECTOR_LENGTH);

    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        printf("%lf ", x[i]);
    }
    printf("\n");

    free(x);

    return 0;
}
