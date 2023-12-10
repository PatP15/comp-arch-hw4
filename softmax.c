#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <stdint.h>
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
        //printf("%lf", x[i]);
	sum += x[i];
    }
    //printf("\n");
    //printf("%lf\n", sum);

    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

__m256d _mm256_exp_pd(__m256d x) {
    __m256d a0 = _mm256_set1_pd(1.0);
    __m256d a1 = _mm256_set1_pd(1.0);
    __m256d a2 = _mm256_set1_pd(1.0 / 2.0);
    __m256d a3 = _mm256_set1_pd(1.0 / 6.0);
    __m256d a4 = _mm256_set1_pd(1.0 / 24.0);
    __m256d a5 = _mm256_set1_pd(1.0 / 120.0);
   // __m256d a6 = _mm256_set1_pd(1.0/ 720.0);

    __m256d x2 = _mm256_mul_pd(x, x);
    __m256d x3 = _mm256_mul_pd(x2, x); 
    __m256d x4 = _mm256_mul_pd(x3, x);
    __m256d x5 = _mm256_mul_pd(x4, x);
    //__m256d x6 = _mm256_mul_pd(x5, x);

    __m256d result = a0;
    result = _mm256_add_pd(result, _mm256_mul_pd(a1, x));
    result = _mm256_add_pd(result, _mm256_mul_pd(a2, x2));
    result = _mm256_add_pd(result, _mm256_mul_pd(a3, x3));
    result = _mm256_add_pd(result, _mm256_mul_pd(a4, x4));
    result = _mm256_add_pd(result, _mm256_mul_pd(a5, x5));
    //result = _mm256_add_pd(result, _mm256_mul_pd(a6, x6));
    return result;
}

void softmax_vectorized(double *x, int size) {
    //find max
    double max_x = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_x) {
            max_x = x[i];
        }
    }

    __m256d sum = _mm256_set1_pd(0.0);
    __m256d max = _mm256_set1_pd(-max_x);
    
    int i;
    for (i = 0; i <= size - 4; i += 4) {
    	__m256d arr = _mm256_load_pd(&x[i]);
	__m256d expArr = _mm256_exp_pd(_mm256_add_pd(arr,max));
	sum = _mm256_add_pd(expArr, sum);
    }

    //remaining
    double remaining = 0.0;
    for(; i<size; i++) {
	x[i] = exp(x[i]-max_x);
	remaining += x[i];
    }
    //printf("remaining sum: %lf\n", remaining);
    //get final sum
    double avx_result[4];
    _mm256_store_pd(avx_result, sum);
    double  sumFinal = 0.0;
    for (int j=0; j<4; j++) {
	sumFinal += avx_result[j];
    }
    sumFinal += remaining;

    //printf("sum final: %lf\n", sumFinal);

    __m256d invSum = _mm256_set1_pd(1.0/sumFinal);

    //store
    for (i = 0; i<= size - 4; i+=4) {
    	__m256d tmp = _mm256_load_pd(&x[i]);
        tmp = _mm256_exp_pd(_mm256_add_pd(tmp, max));
	tmp = _mm256_mul_pd(invSum, tmp);
	_mm256_store_pd(&x[i], tmp);	
    }

    for (;i<size;i++) {
    	x[i]/=sumFinal;
    }

}

static inline uint64_t rdtsc(void) {

    unsigned int lo, hi;
    asm volatile("mfence");
    asm volatile("rdtsc": "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

void randVector(double* x)
{
     // Initialize vector x with random values
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        x[i] = (double)rand() / RAND_MAX;
    }
}

int main() {
    
    double *x = malloc(VECTOR_LENGTH * sizeof(double));
    double *y = malloc(VECTOR_LENGTH * sizeof(double));

    randVector(x);
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        printf("%lf ", x[i]);
    }
    printf("\n");

    
    int repeats = 10;
    // Serial softmax computation

    for(int i = 0; i < repeats; i++)
    {
        randVector(x);
        uint64_t start = rdtsc();
        softmax(x, VECTOR_LENGTH);
        uint64_t end = rdtsc();
        printf("Serial time taken: %lu cycles\n", end - start);
    }
   
    // Vector softmax computation
   
    for(int i = 0; i < repeats; i++)
    {
        randVector(y);
        uint64_t start = rdtsc();
        softmax_vectorized(y, VECTOR_LENGTH);
        uint64_t end = rdtsc();
        printf("Vector time taken: %lu cycles\n", end - start);
    }
   
    
    
   /*
    double *x = malloc(VECTOR_LENGTH * sizeof(double));
    double *y = malloc(VECTOR_LENGTH * sizeof(double));

    // Initialize vector x with random values
    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        x[i] = (double)rand() / RAND_MAX;
    	y[i] = x[i];
    }

    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        printf("%lf ", x[i]);
    }
    printf("\n");

    // Softmax computation
    softmax(x, VECTOR_LENGTH);
    softmax_vectorized(y, VECTOR_LENGTH);

    for (int i = 0; i < VECTOR_LENGTH; ++i) {
        printf("%lf ", x[i]);
    }
    printf("\n");

    for (int i = 0; i < VECTOR_LENGTH; ++i) {
	    printf("%lf ", y[i]);
    }
    printf("\n");

    free(x);
    free(y);*/
    free(x);
    free(y);
    return 0;
}
