#include <stdio.h>
#include <stdlib.h>

#define NUM_WORDS 20
#define EMBEDDING_DIM 5

void embeddingLookup(double *embedding_matrix, int *indices, double *embeddings, int num_indices) {
    for (int i = 0; i < num_indices; ++i) {
        int index = indices[i];
        for (int j = 0; j < EMBEDDING_DIM; ++j) {
            embeddings[i * EMBEDDING_DIM + j] = embedding_matrix[index * EMBEDDING_DIM + j];
        }
    }
}

int main() {
    double *embedding_matrix = malloc(NUM_WORDS * EMBEDDING_DIM * sizeof(double));
    int *indices = malloc(20 * sizeof(int));
    double *embeddings = malloc(20 * EMBEDDING_DIM * sizeof(double));

    // Initialize embedding matrix with random values
    for (int i = 0; i < NUM_WORDS * EMBEDDING_DIM; ++i) {
        embedding_matrix[i] = i;
    }

    // Initialize indices (word IDs)
    for (int i = 0; i < 20; ++i) {
        indices[i] = rand() % NUM_WORDS;
	printf("%d ", indices[i] * EMBEDDING_DIM);
    }
    printf("\n\n");

    // Embedding lookup
    embeddingLookup(embedding_matrix, indices, embeddings, 20);

    for (int i = 0; i < 20 * EMBEDDING_DIM; ++i) {
        printf("%lf ", embeddings[i]);
	if (i % EMBEDDING_DIM == 4 && i != 0) {
	    printf("\n");
	}
    }

    free(embedding_matrix);
    free(indices);
    free(embeddings);

    return 0;
}
