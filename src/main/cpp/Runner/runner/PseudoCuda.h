#pragma once
#include <string>
#include <chrono>
#include <ctime>  

struct dim3 {
	size_t x;
	size_t y;
	size_t z;
	dim3() : x(), y(), z() {}
	dim3(size_t x) : x(x), y(1), z(1) {}
	dim3(size_t x, size_t y) : x(x), y(y), z(1) {}
	dim3(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {}
	std::string operator+(const std::string& other) const {
		return std::string("(") + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")" + other;
	}
};

std::string operator+(const std::string& s, const dim3& d); 

int cudaMalloc(void** ptr, size_t size);
int cudaMemcpy(void * dst, void* src, size_t size, int direction);
int cudaFree(void* ptr);
int cudaMemset(void* ptr, int val, size_t count);


#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 0
#define __global__
#define __device__


extern dim3 threadId;
extern dim3 blockId;
extern dim3 gridDim;
extern dim3 blockDim;

typedef std::chrono::system_clock::time_point cudaEvent_t;

int cudaEventCreate(cudaEvent_t* t);
int cudaEventRecord(cudaEvent_t& t, int a);
int cudaEventSyncronize(cudaEvent_t t);
int cudaEventDestroy(cudaEvent_t t);
int cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);