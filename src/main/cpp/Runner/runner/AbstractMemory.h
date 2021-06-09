#pragma once

class AbstractMemory
{
protected:
	void* initMultiarray(size_t* dimensions, size_t left, size_t size) {
		if (left == 0) {
			void* res = new char[size * (*dimensions)];
			memset(res, 0, size * (*dimensions));
			return res;
		}
		else {
			void** res = new void* [*dimensions];
			for (size_t i = 0; i < *dimensions; i++) {
				res[i] = initMultiarray(dimensions + 1, left - 1, size);
			}
			return res;
		}
	}
	void deleteMultiarray(void** ptr, size_t* dimensions, size_t left) {
		if (left == 0) {
			delete[] ptr;
		}
		else {
			for (size_t i = 0; i < *dimensions; i++)
				deleteMultiarray((void**)ptr[i], dimensions + 1, left - 1);
			delete[] ptr;
		}
	}

public:
	virtual void* getPtr(size_t index) { return nullptr; };
};

