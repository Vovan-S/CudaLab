#pragma once
#include "../runner/Thread.h"
#include "counter.h"

class memory__count_figures_gpu : public AbstractMemory {
public:
	memory__count_figures_gpu() {}
	void* getPrt(size_t) { return nullptr; }
};

class global__count_figures_gpu : public Thread {
	PlanePart* plane;
	CountStruct* count;
	CountAlgConfig alg_cfg;
	size_t tid;
	size_t x;
	size_t y;
	size_t dim;
	size_t i;
	size_t x1;
	size_t y1;
	plane_t color;
	char* visited;
	LineConfig cfg;
	SmallQuery sq;
	char flag;

	 AbstractMemory* _shared;

public:
	global__count_figures_gpu(PlanePart* plane, CountStruct* count, CountAlgConfig alg_cfg):
		_shared(nullptr),
		plane(plane),
		count(count),
		alg_cfg(alg_cfg),
		tid(),
		x(),
		y(),
		dim(),
		i(),
		x1(),
		y1(),
		color(),
		visited(),
		cfg(),
		sq(),
		flag() {}

	bool usingShared() { return false; }

	AbstractMemory* buildSharedMemory() { return new memory__count_figures_gpu(); }

	Thread* build(dim3 threadId, AbstractMemory* shared) {
		global__count_figures_gpu* new_thread = new global__count_figures_gpu(plane,
		                                                                      count,
		                                                                      alg_cfg);
		new_thread->_shared = shared;
		new_thread->m_threadId = threadId;
		return new_thread;
	}

	int run() {

switch(current_part) {
case 0: goto enter; 
default: return -1;
}
enter:

    // func starts
    tid = threadId.x + blockId.x * blockDim.x;
    dim = gridDim.x * blockDim.x;
    
    while(tid < 2 * alg_cfg.k * (alg_cfg.k - 1)) {
        cudaMemset(visited, 0, sizeof(visited));
        cfg = map(tid, &alg_cfg);
        for (i = 0; i < alg_cfg.len; i++) {
            if (0 == cfg.direction) { // горизонтальная граница
                x = cfg.start + i;
                y = cfg.cy;
                x1 = x;
                y1 = y - 1;
            }
            else {
                x = cfg.cx;
                y = cfg.start + i;
                x1 = x - 1;
                y1 = y;
            }
            if (!visited[i]) {
                color = pixel(plane, x, y);
                if (TRIAG_COLOR != color && CIRCLE_COLOR != color) {
                    color = pixel(plane, x1, y1);
                }
                if (TRIAG_COLOR == color || CIRCLE_COLOR == color) {
                    sq_push(&sq, x, y);
                    visited[i] = 1;
                    flag = 1;
                    while (traverse(plane, &sq, alg_cfg.len << cfg.rang, &x, &y, color)) {
                        if (y == cfg.cy && x < cfg.stop) visited[x - cfg.start] = 1;
                        if (!should_count(x, y, &cfg, alg_cfg.len)) {
                            flag = 0;
                            break;
                        }
                    }
                    if (flag) {
                        if (TRIAG_COLOR == color) {
                            count->triag[tid] += 1;
                        } else {
                            count->circle[tid] += 1;
                        }
                    }
                }
            }
            
        }
        tid += dim;
    }
    return 0;   

	}
};
