#pragma once
void CPU_mulMatrix_serial(int* outMatrix, int* inMatrixA, int* inMatrixB, int halfSize);
void CPU_mulMatrix_parallel(int* outMatrix, int* inMatrixA, int* inMatrixB, int halfSize);
