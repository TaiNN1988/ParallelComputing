#pragma once
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include <iostream>
#include <chrono>

using namespace std;

void showMatrix(int* matrix, int halfSize);
bool compareMatrix(int* A, int* B, int halfSize);