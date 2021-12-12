__kernel void mulMatrix(
	__global int* outMatrix,
	__global const int* inMatrix1,
	__global const int* inMatrix2,
	int halfSize
) {
	// Thread identifiers
	const int row = get_global_id(0);
	const int col = get_global_id(1);
	//int halfSize = get_global_size(0);

	if (row < halfSize && col < halfSize)
	{
		int sum = 0;
		for (int k = 0; k < halfSize; k++) {
			sum += inMatrix1[k+halfSize*row] * inMatrix2[col + halfSize*k];
		}

		outMatrix[col + row * halfSize] = sum;
	}
}