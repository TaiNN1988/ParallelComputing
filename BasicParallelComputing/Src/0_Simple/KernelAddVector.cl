__kernel void addVectors(__global int* d_outVector,
	__global const int* d_inVector1,
	__global const int* d_inVector2) {
	int idx = get_global_id(0);
	int size = get_global_size(0);
	
	if(idx < size)
		d_outVector[idx] = d_inVector1[idx] + d_inVector2[idx];
}