#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <cuda.h>

#include "support.h"
#include "kernel.cu"

// to activate debug statements
#define DEBUG 1

// program constants
#define BLOCK_SIZE 1024

// solution constants

//functions used
void err_check(cudaError_t ret, char* msg, int exit_code);


#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

#define BLUR_SIZE 2

int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_serial n_row n_col mat_input.csv mat_output.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);

    // Get files to read/write 
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[2]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    // Matrices to use
    int* filterMatrix_h = (int*)malloc(5 * 5 * sizeof(int));
    int* inputMatrix_h  = (int*) malloc(n_row * n_col * sizeof(int));
    int* outputMatrix_h = (int*) malloc(n_row * n_col * sizeof(int));

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            inputMatrix_h[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }


    // Filling filter
	// 1 0 0 0 1 
	// 0 1 0 1 0 
	// 0 0 1 0 0 
	// 0 1 0 1 0 
	// 1 0 0 0 1 
    for(int i = 0; i< 5; i++)
        for(int j = 0; j< 5; j++)
            filterMatrix_h[i*5+j]=0;

    filterMatrix_h[0*5+0] = 1;
    filterMatrix_h[1*5+1] = 1;
    filterMatrix_h[2*5+2] = 1;
    filterMatrix_h[3*5+3] = 1;
    filterMatrix_h[4*5+4] = 1;
    
    filterMatrix_h[4*5+0] = 1;
    filterMatrix_h[3*5+1] = 1;
    filterMatrix_h[1*5+3] = 1;
    filterMatrix_h[0*5+4] = 1;

    fclose(inputFile1); 


	Timer timer;
	startTime(&timer);
	cudaError_t cuda_ret;
	int num_blocks = ceil((float)(n_row * n_col) / (float)BLOCK_SIZE);
    dim3 dimGrid(num_blocks, 1, 1);
	dim3 dimBlock(BLOCK_SIZE, 1, 1);	
    // --------------------------------------------------------------------------- //
    // ------ Algorithm Start ---------------------------------------------------- //

	int* d_input;
	int* d_filter;
	int* d_output;
	cuda_ret = cudaMalloc((void**)&d_input, n_row * n_col * sizeof(int));
	err_check(cuda_ret, (char*)"Unable to allocate memory to device!", 1);
	cuda_ret = cudaMalloc((void**)&d_filter, 5 * 5 * sizeof(int));
	err_check(cuda_ret, (char*)"Unable to allocate memory to device!", 1);
	cuda_ret = cudaMalloc((void**)&d_output, n_row * n_col * sizeof(int));
	err_check(cuda_ret, (char*)"Unable to allocate memory to device!", 1);

    struct timespec start, end;    
    clock_gettime(CLOCK_REALTIME, &start);
	cuda_ret = cudaMemcpy(d_input, inputMatrix_h, n_row * n_col * sizeof(int), cudaMemcpyHostToDevice);
	err_check(cuda_ret, (char*)"Unable to copy memory to device!", 1);
	cuda_ret = cudaMemcpy(d_filter, filterMatrix_h, 5 * 5 * sizeof(int), cudaMemcpyHostToDevice);
	err_check(cuda_ret, (char*)"Unable to copy memory to device!", 1);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_copyin = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;


    clock_gettime(CLOCK_REALTIME, &start);
	conv_kernel <<< dimGrid, dimBlock >>> (
			d_input,
			d_filter,
			d_output,
			n_row,
			n_col,
			BLUR_SIZE);
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch nonce kernel!", 2);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_conv = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

	int* d_pool;	
	cuda_ret = cudaMalloc((void**)&d_pool, n_row * n_col * sizeof(int));
	err_check(cuda_ret, (char*)"Unable to allocate memory to device!", 1);

    clock_gettime(CLOCK_REALTIME, &start);
	maxpool_kernel <<< dimGrid, dimBlock >>> (
			d_output,
			d_pool,
			n_row,
			n_col,
			BLUR_SIZE);
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch nonce kernel!", 2);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_pool = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    clock_gettime(CLOCK_REALTIME, &start);
	cuda_ret = cudaMemcpy(outputMatrix_h, d_pool, n_row * n_col * sizeof(int), cudaMemcpyDeviceToHost);
	err_check(cuda_ret, (char*)"Unable to copy memory to host!", 1);
    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent_copyout = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    // --------------------------------------------------------------------------- //
    // ------ Algorithm End ------------------------------------------------------ //


	// Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", outputMatrix_h[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Print time
    fprintf(timeFile, "%.20f", time_spent_copyin);
    fprintf(timeFile, "%.20f", time_spent_conv);
    fprintf(timeFile, "%.20f", time_spent_pool);
    fprintf(timeFile, "%.20f", time_spent_copyout);

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

	cudaFree(d_input);
	cudaFree(d_filter);
	cudaFree(d_output);
	cudaFree(d_pool);
    free(inputMatrix_h);
    free(outputMatrix_h);
    free(filterMatrix_h);

    return 0;
}

/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //
