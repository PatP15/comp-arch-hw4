#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdint.h>
#include <xmmintrin.h> 


#define SIZE 2049

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



void transposeMatrix(double src[SIZE][SIZE], double dest[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            dest[j][i] = src[i][j];
        }
    }
}

void vecMatrixMultNoTranspose(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    int idx = 0;

    for (int i = 0; i < SIZE; i++) {
         //4 0s
        
        idx = 0;
        //int c = 4;
        for (int c = 0; c < SIZE; c+=4){
            //loop through the column group of the 4s
            __m256d sum = _mm256_setzero_pd();
            for (int j = 0; j < SIZE; j++) {
                //copy the element into the entire array
                __m256d a_elem = _mm256_broadcast_sd(&A[i][j]);

                //get the column values from B
                __m256d b_col = _mm256_set_pd(B[j][c+3], B[j][c+2], B[j][c+1], B[j][c]);

                //multiply and add
                sum = _mm256_add_pd(sum, _mm256_mul_pd(a_elem, b_col));
            }
            //store the result in C
            
            _mm256_storeu_pd(&C[i][idx], sum);
            idx += 4;
        }
        
    }
    //do the remaining columns serially
    if (SIZE % 4 != 0) {
        for (int i = SIZE - SIZE % 4; i < SIZE; i++) {
            for (int c = SIZE - SIZE % 4; c < SIZE; c++) {
                double sum = 0.0;
                for (int j = 0; j < SIZE; j++) {
                    sum += A[i][j] * B[j][c];
                }
                C[i][c] = sum;
            }
        }
    }
}


void vecMatrixMultTranspose(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
    int idx = 0;

    double (*B_tran)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*B_tran), 32);

    transposeMatrix(B, B_tran);

    // printf("transposed\n");
    // printMatrix(B_tran);
    for (int i = 0; i < SIZE; i++) {
         //4 0s
        
        idx = 0;
        //int c = 4;

        //loop through the column group of the 4s

        for (int j = 0; j < SIZE; j++) {
            __m256d sum = _mm256_setzero_pd();
            
            double total_sum = 0;
            for(int k = 0; k < SIZE; k+=4){
                //copy the element into the entire array
                if((k+3) < SIZE)
                {
                    // printf("i: %d\n", i);
                    // printf("j: %d\n", j);
                    // printf("k: %d\n", k);
                    // printf("A: %f\n", A[j][k]);
                    __m256d b_col = _mm256_loadu_pd(&B_tran[i][k]);
                    // printf("not here\n");
                    // printMatrix(A);
                    __m256d a_row = _mm256_loadu_pd(&A[j][k]);
                    // printf("not here\n");
                    
                    double temp[4];

                    // Store the contents of a_row into temp and print
                    // _mm256_storeu_pd(temp, a_row);
                    // printf("a_row: %f %f %f %f\n", temp[0], temp[1], temp[2], temp[3]);

                    // Store the contents of b_col into temp and print
                    // _mm256_storeu_pd(temp, b_col);
                    // printf("b_col: %f %f %f %f\n", temp[0], temp[1], temp[2], temp[3]);
                    //multiply and add

                    __m256d mult = _mm256_mul_pd(a_row, b_col);

                    _mm256_storeu_pd(temp, mult);
                    // printf("mult: %f %f %f %f\n", temp[0], temp[1], temp[2], temp[3]);

                    // sum = _mm256_add_pd(sum, mult);
                    __m256d sum_pair = _mm256_hadd_pd(mult, mult);

                    
                    __m128d low = _mm256_extractf128_pd(sum_pair, 0);
                    __m128d high = _mm256_extractf128_pd(sum_pair, 1);

                    // dd two __m128d vectors
                    __m128d final_sum = _mm_add_pd(low, high);

                    double result[2]; //only need first element, but _mm_storeu_pd requires two
                    _mm_storeu_pd(result, final_sum);
                    
                    total_sum += result[0];
                }
                // printf("Total Sum: %f\n", total_sum);
            }
            //store the result in C
            
            if (SIZE % 4 != 0) {
                // printf("leftovers\n");
                for (int k = SIZE - SIZE % 4; k < SIZE; k++) {
                    // printf("A: %f * B: %f\n", A[i][k], B_tran[j][k]);
                    total_sum += A[j][k] * B_tran[i][k];
                }
                // printf("Sum: %f\n", total_sum);
            }

            C[j][i] = total_sum; 
            // printf("Sum: %f\n", C[j][i]);
            // break;
        }
        
       
    }
    // do the remaining columns serially
    // if (SIZE % 4 != 0) {
    //     for (int i = SIZE - SIZE % 4; i < SIZE; i++) {
    //         for (int c = SIZE - SIZE % 4; c < SIZE; c++) {
    //             double sum = 0.0;
    //             for (int j = 0; j < SIZE; j++) {
    //                 sum += A[i][j] * B[j][c];
    //             }
    //             C[i][c] = sum;
    //         }
    //     }
    // }
    _mm_free(B_tran);
}



int min(int a, int b) {
    return (a < b) ? a : b;
}


#define CACHEBLOCK 256
// void vecMatrixMultCacheBlocking(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
// void vecMatrixMultTranspose(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
//     int idx = 0;

//     double (*B_tran)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*B_tran), 32);

//     transposeMatrix(B, B_tran);

//     // printf("transposed\n");
//     // printMatrix(B_tran);
//     for (int i = 0; i < SIZE; i++) {
//          //4 0s
        
//         idx = 0;
//         //int c = 4;

//         //loop through the column group of the 4s

//         for (int j = 0; j < SIZE; j++) {
//             __m256d sum = _mm256_setzero_pd();
            
//             double total_sum = 0;
//             for(int k = 0; k < SIZE; k+=4){
//                 //copy the element into the entire array
//                 if((k+3) < SIZE)
//                 {
//                     // printf("i: %d\n", i);
//                     // printf("j: %d\n", j);
//                     // printf("k: %d\n", k);
//                     // printf("A: %f\n", A[j][k]);
//                     __m256d b_col = _mm256_load_pd(&B_tran[i][k]);
//                     // printf("not here\n");
//                     // printMatrix(A);
//                     __m256d a_row = _mm256_load_pd(&A[j][k]);
//                     // printf("not here\n");

//                     __m256d mult = _mm256_mul_pd(a_row, b_col);

//                     _mm256_storeu_pd(temp, mult);
//                     // printf("mult: %f %f %f %f\n", temp[0], temp[1], temp[2], temp[3]);

//                     // sum = _mm256_add_pd(sum, mult);
//                     __m256d sum_pair = _mm256_hadd_pd(mult, mult);

                    
//                     __m128d low = _mm256_extractf128_pd(sum_pair, 0);
//                     __m128d high = _mm256_extractf128_pd(sum_pair, 1);

//                     // dd two __m128d vectors
//                     __m128d final_sum = _mm_add_pd(low, high);

//                     double result[2]; //only need first element, but _mm_storeu_pd requires two
//                     _mm_storeu_pd(result, final_sum);
                    
//                     total_sum += result[0];
//                 }
//                 // printf("Total Sum: %f\n", total_sum);
//             }
//             //store the result in C
            
//             if (SIZE % 4 != 0) {
//                 // printf("leftovers\n");
//                 for (int k = SIZE - SIZE % 4; k < SIZE; k++) {
//                     // printf("A: %f * B: %f\n", A[i][k], B_tran[j][k]);
//                     total_sum += A[j][k] * B_tran[i][k];
//                 }
//                 // printf("Sum: %f\n", total_sum);
//             }

//             C[j][i] = total_sum; 
//         }
//     }
//     _mm_free(B_tran);
// }

static inline uint64_t rdtsc(void) {

    unsigned int lo, hi;
    asm volatile("mfence");
    asm volatile("rdtsc": "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

void printMatrix(double matrix[SIZE][SIZE]){
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            printf("%lf ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    // SIZE = atoi(argv[1]);

    printf("The matrices are of shape %d x %d\n", SIZE, SIZE);
    // Allocate memory for matrices A, B, and C
    double (*A)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*A), 32);
    double (*B)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*B), 32);
    double (*C)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*C), 32);
    double (*D)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*D), 32);
    double (*E)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*E), 32);
    double (*F)[SIZE] = (double (*)[SIZE])_mm_malloc(SIZE * sizeof(*F), 32);

    // Initialize matrices A and B with random values
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            //A[i][j] = (double)rand() / RAND_MAX;
            //B[i][j] = (double)rand() / RAND_MAX;
            A[i][j] = i+2;
            B[i][j] = j+8;
        }
    }
    // printMatrix(A);
    // printf("\n\n");
    // printMatrix(B);
    // for (int i = 0; i < SIZE; ++i) {
    //     for (int j = 0; j < SIZE; ++j) {
    //         printf("%lf ", A[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // for (int i = 0; i < SIZE; ++i) {
    //     for (int j = 0; j < SIZE; ++j) {
    //         printf("%lf ", B[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // Perform matrix multiplication


    int repeat = 10;
   
    printf("\nSerialized\n");

    for(int i = 0; i < repeat; i++){
        uint64_t start = rdtsc();
        serialMatrixMult(A, B, C);
        uint64_t end = rdtsc();
        printf("Serial time taken: %lu cycles\n", end - start);
    }
    // printMatrix(C);
    printf("\nVectorized\n");

    for(int i = 0; i < repeat; i++){
        uint64_t start = rdtsc();
        vecMatrixMultNoTranspose(A, B, D);
        uint64_t end = rdtsc();
        printf("Vector time taken: %lu cycles\n", end - start);
    }
   
    // printMatrix(D);
    printf("\nVectorized w/ Transpose\n");

    for(int i = 0; i < repeat; i++){
        uint64_t start = rdtsc();
        vecMatrixMultTranspose(A, B, E);
        uint64_t end = rdtsc();
        printf("Vector time taken: %lu cycles\n", end - start);
    }

    // printMatrix(E);
    // printf("\nVectorized w/Transpose and Cache Blocking\n");

    // for(int i = 0; i < repeat; i++){
    //     uint64_t start = rdtsc();
    //     vecMatrixMultCacheBlocking(A, B, F);
    //     uint64_t end = rdtsc();
    //     printf("Vector time taken: %lu cycles\n", end - start);
    // }

   
    // printMatrix(Z);

     

    for (int i = 0; i < SIZE; ++i) {
        for (int j =0; j < SIZE; ++j) {
            //A[i][j] = (double)rand() / RAND_MAX;
            //B[i][j] = (double)rand() / RAND_MAX;
            if(D[i][j] != C[i][j] || E[i][j] != C[i][j])
            {
                
                printf("I: %d, J: %d", i, j);
                //break;
            }
        }
    }
    
    
    

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(D);
    _mm_free(E);
    return 0;
}
