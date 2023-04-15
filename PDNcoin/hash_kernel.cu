
/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/
__global__
void hash_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int* transactions, unsigned int n_transactions, unsigned int mod) {

    // Calculate thread index
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // TODO: Generate hash values
	if (index < array_size) {
		unsigned int hash = (nonce_array[index] + transactions[0] * (index + 1)) % mod;
		for (int j = 1; j < n_transactions; j ++){
			hash = (hash + transactions[j] * (index + 1)) % mod;
		}
		hash_array[index] = hash;
	}

} // End Hash Kernel //
