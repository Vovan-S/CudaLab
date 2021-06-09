#pragma once
#include "../runner/Thread.h"
#include "counter.h"

class memory__reduction : public AbstractMemory {
public:
	memory__reduction() {}
	void* getPrt(size_t) { return nullptr; }
};

class global__reduction : public Thread {
	CountStruct* count;
	FigureCount* fc;
	size_t elems;
	size_t tid;
	size_t dim;
	size_t i;
	size_t j;

	 AbstractMemory* _shared;

public:
	global__reduction(CountStruct* count, FigureCount* fc, size_t elems):
		_shared(nullptr),
		count(count),
		fc(fc),
		elems(elems),
		tid(),
		dim(),
		i(),
		j() {}

	bool usingShared() { return false; }

	AbstractMemory* buildSharedMemory() { return new memory__reduction(); }

	Thread* build(dim3 threadId, AbstractMemory* shared) {
		global__reduction* new_thread = new global__reduction(count,
		                                                      fc,
		                                                      elems);
		new_thread->_shared = shared;
		new_thread->m_threadId = threadId;
		return new_thread;
	}

	int run() {

switch(current_part) {
case 0: goto enter; 
case 1: goto sync1;
default: return -1;
}
enter:

    // func starts
    if (blockId.x != 0) return 0;
    tid = threadId.x;
    dim = gridDim.x;
    
    for (i = 0; tid + i * dim < elems; i++) {
        count->circle[tid] += count->circle[tid + i * dim];
        count->triag[tid] += count->triag[tid + i * dim];
    }
    
    i = dim / 2;
    while (i != 0) {
        while (tid < i && tid + i < elems) {
            count->circle[tid] += count->circle[tid + i];
            count->triag[tid] += count->triag[tid + i];
        }
        current_part = 1; return 1;
sync1: __asm nop

        i /= 2;
    }
    fc->circles = count->circle[0];
    fc->triags = count->triag[0];
    return 0;

	}
};
