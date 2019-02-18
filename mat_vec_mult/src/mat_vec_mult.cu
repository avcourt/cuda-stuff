#include <stdio.h>
#include <stdlib.h>

/* COMP 4510-A3Q2
 *
 *  Matrix vector multiplciation in CUDA
 *
 *  ANDREW VAILLANCOURT
 *  7729503
 */

#define N 16

 // PROTOTYPES
void init_vec(float *vec, int len);
void print_vec(const char *label, float *vec, int len);
void signoff();
__global__ void mult(float *a, float *b, int n);
__global__ void reduce(float *input, float *output);


// MAIN
int main(int argc, char *argv[])
{
	const int n = N;
	int a_len = n * n;

    // Allocate host buffers
    size_t size = n * sizeof(float); 			// size in bytes
    float *host_a = (float *) malloc(size * n); // NxN matrix A
    float *host_b = (float *) malloc(size);		// b vector
    float *host_c = (float *) malloc(size);		// c vector

    // Fill host buffers with floats and print them
    init_vec(host_a, a_len);
    init_vec(host_b, n);
    print_vec("Matrix A:\n", host_a, a_len);
    print_vec("Vector B:\n", host_b, n);

    // Allocate device buffers
    float *dev_a;
    float *dev_b;
    float *dev_c;
    cudaMalloc(&dev_a, size * n);
    cudaMalloc(&dev_b, size);
    cudaMalloc(&dev_c, size * n);

    // Transfer matrix_A and vec_b from host to device
    cudaMemcpy(dev_a, host_a, size * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size, cudaMemcpyHostToDevice);

    // call multiply kernel
    int threads = n * n;	// one thread for each element of A
    mult<<<1, threads>>>(dev_a, dev_b, n);

    // Transfer the resulting A vector back to the host to print
    cudaMemcpy(host_a, dev_a, size * n, cudaMemcpyDeviceToHost);
    print_vec("Matrix A(after multiply operation):\n", host_a, a_len);

    // now row reduce, Matrix A is still resident on GPU, no need to transfer A again
    reduce<<<n, n/2>>>(dev_a, dev_c);
    cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);
    print_vec("c vector ( A*b = c ):\n", host_c, n );

    // Clean up memory on host
    free(host_a);
    free(host_b);
    free(host_c);

    // Clean up memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    signoff();
    return EXIT_SUCCESS;
}

// KERNELS

// multiplies each element of matrix a by vector b in place
__global__ void mult(float *a, float *b, int n)
{
    // determine this thread's global id
    int global_id = threadIdx.x;
    // If we have extra threads running in the last block, make sure they don't add anything!
    if (global_id < n * n)
    {
        a[global_id] = a[global_id] * b[global_id % n];
    }
}

// sums the N rows of input, returns them in output vector of size N
__global__ void reduce(float *input, float *output)
{
    unsigned int thread_id = threadIdx.x;
    unsigned int block_id = blockIdx.x;

    // The size of the chunk of data this thread's block is working on.
    unsigned int chunk_size = N;

    // Calculate the index that this block's chunk of values starts at.
    // Each thread adds 2 values, so each block adds a total of block_size * 2 values.
    unsigned int block_start = block_id * chunk_size;

    unsigned int left;  // holds index of left operand
    unsigned int right; // holds index or right operand
    unsigned int threads = N / 2;
    for (unsigned int stride = 1; stride < chunk_size; stride *= 2, threads /= 2)
    {
        left = block_start + thread_id * (stride * 2);
        right = left + stride;

        if (thread_id < threads )
        {
            input[left] += input[right];
        }
        __syncthreads();
    }

    if (!thread_id)
    {
        output[block_id] = input[block_start];
    }
}

// MATRIX/VECTOR UTILITY FUNCTIONS
// init vectors w/ random flaots
void init_vec(float *vec, int len)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand(time(NULL));
        seeded = 1;
    }

    int i;
    for (i = 0; i < len; i++)
    {
        vec[i] = (float) rand() / RAND_MAX;
    }
}

// Prints the given vector to stdout
void print_vec(const char *label, float *vec, int len)
{
    printf("%s", label);

    int i;
    for (i = 0; i < len; i++)
    {
    	if(i % N == 0) printf("\n");
        printf("%f ", vec[i]);
    }
    printf("\n\n");
}

void signoff() {
    printf("\n ====================== Processing Complete ==========================\n");
    printf(" =============== Programmed by Andrew Vaillancourt ===================\n");
    printf(" =====================================================================\n");
}
