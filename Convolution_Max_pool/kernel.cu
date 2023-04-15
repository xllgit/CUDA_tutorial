

/* Conv Kernel ----------------------------------
*       Generates an array of conv values.
*/
__global__
void conv_kernel(int* input, int* filter, int* output, int n_row, int n_col, int BLUR_SIZE) {

    // Calculate thread rank
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x; 
	if (index < n_row * n_col){
		int i = index / n_row;
		int j = index % n_row;
		int sum_val = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow)
        {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
            {
                int curRow = i + blurRow;
                int curCol = j + blurCol;

                int i_row = blurRow + BLUR_SIZE;
                int i_col = blurCol + BLUR_SIZE;

                if( curRow > -1 && curRow < n_row && curCol > -1 && curCol < n_col)
                {
                    sum_val += input[curRow*n_col + curCol]*filter[i_row*5 + i_col]; 
                }
            }
        }

        output[index] = sum_val;
	}

} // End Conv Kernel //

/* max pooling */
__global__
void maxpool_kernel(int* input, int* output, int n_row, int n_col, int BLUR_SIZE) {

    // Calculate thread rank
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x; 
	if (index < n_row * n_col){
		int i = index / n_row;
		int j = index % n_row;
		int max = 0;
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow)
        {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol)
            {
                int curRow = i + blurRow;
                int curCol = j + blurCol;

                if( curRow > -1 && curRow < n_row && curCol > -1 && curCol < n_col)
                {
					if (input[curRow * n_col + curCol] > max)
						max = input[curRow * n_col + curCol];
                }
            }
        }

        output[index] = max;
	}

} // End Conv Kernel //

