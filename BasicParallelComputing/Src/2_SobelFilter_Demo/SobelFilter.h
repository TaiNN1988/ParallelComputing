#pragma once
int CPU_sobelFilter_serial(unsigned char* outImage, unsigned char* srcImage, int width, int height);
int CPU_sobelFilter_parallel(unsigned char* outImage, unsigned char* srcImage, int width, int height);