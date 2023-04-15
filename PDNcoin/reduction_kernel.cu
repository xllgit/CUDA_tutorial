
/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/
__device__ void warpReduce(volatile float *sdata, unsigned int tid) 
  {
      sdata[tid] = min(sdata[tid],sdata[tid+32]);
      sdata[tid] = min(sdata[tid],sdata[tid+16]);
      sdata[tid] = min(sdata[tid],sdata[tid+8]);
      sdata[tid] = min(sdata[tid],sdata[tid+4]);
      sdata[tid] = min(sdata[tid],sdata[tid+2]);
      sdata[tid] = min(sdata[tid],sdata[tid+1]);
      
  }  

__global__
void reduction_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int MAX) {
    
    unsigned int unique_id = blockIdx.x * blockDim.x + threadIdx.x; /* unique id for each thread in the block*/

    unsigned int thread_id = threadIdx.x; /* thread index in the block*/

	extern __shared__ int total[];
    int* minChunk = total;
    int* minNonce = &total[blockDim.x];

    if(unique_id < array_size){
        minChunk[thread_id] = hash_array[unique_id];
        minNonce[thread_id] = nonce_array[unique_id];
    }
	else{
        minChunk[thread_id] = MAX;
        minNonce[thread_id] = MAX;
	}


    __syncthreads();

    for(unsigned int stride = (blockDim.x/2); stride > 0 ; stride /=2){
        __syncthreads();

        if(thread_id < stride)
        {
			if(minChunk[thread_id] > minChunk[thread_id + stride]){
				minChunk[thread_id] = minChunk[thread_id + stride];
				minNonce[thread_id] = minNonce[thread_id + stride];
			}
        }
    }

    //if(thread_id < 32){
    //    warpReduce(minChunk,thread_id);
    //}

    if(thread_id == 0){
		hash_array[blockIdx.x] = minChunk[0];
		nonce_array[blockIdx.x] = minNonce[0];
    }
} // End reduce Kernel //
