#include "PseudoCuda.h"

dim3 threadId;
dim3 blockId;
dim3 gridDim;
dim3 blockDim;

std::string operator+(const std::string& s, const dim3& d) {
		return s + (d + std::string());	return std::string();
}

int cudaMalloc(void** ptr, size_t size)
{
	*ptr = new char[size];
	return 0;
}

int cudaMemcpy(void* dst, void* src, size_t size, int)
{
	memcpy(dst, src, size);
	return 0;
}

int cudaFree(void* ptr)
{
	delete[] ptr;
	return 0;
}

int cudaMemset(void* ptr, int val, size_t count)
{
	memset(ptr, val, count);
	return 0;
}

int cudaEventCreate(cudaEvent_t* t)
{
	return 0;
}

int cudaEventRecord(cudaEvent_t& t, int a)
{
	t = std::chrono::system_clock::now();
	return 0;
}

int cudaEventSyncronize(cudaEvent_t t)
{
	return 0;
}

int cudaEventDestroy(cudaEvent_t t)
{
	return 0;
}

int cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end)
{
	std::chrono::duration<float> elasped_seconds = end - start;
	return 1000 * elasped_seconds.count();
}
