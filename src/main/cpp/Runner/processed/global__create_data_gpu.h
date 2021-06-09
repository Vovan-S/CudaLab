#pragma once
#include "../runner/Thread.h"
#include "datacreation.h"

class memory__create_data_gpu : public AbstractMemory {
public:
	memory__create_data_gpu() {}
	void* getPrt(size_t) { return nullptr; }
};

class global__create_data_gpu : public Thread {
	PlanePart* plane;
	CreationSettingsStruct* settings;
	CreationControlStruct* control;
	FigureCount* actual_count;
	size_t tid;
	size_t generated;
	size_t i;
	size_t dim;
	size_t seed;
	Circle * p_current;
	Circle * p_check;

	 AbstractMemory* _shared;

public:
	global__create_data_gpu(PlanePart* plane, CreationSettingsStruct* settings, CreationControlStruct* control, FigureCount* actual_count):
		_shared(nullptr),
		plane(plane),
		settings(settings),
		control(control),
		actual_count(actual_count),
		tid(),
		generated(),
		i(),
		dim(),
		seed(),
		p_current(),
		p_check() {}

	bool usingShared() { return false; }

	AbstractMemory* buildSharedMemory() { return new memory__create_data_gpu(); }

	Thread* build(dim3 threadId, AbstractMemory* shared) {
		global__create_data_gpu* new_thread = new global__create_data_gpu(plane,
		                                                                  settings,
		                                                                  control,
		                                                                  actual_count);
		new_thread->_shared = shared;
		new_thread->m_threadId = threadId;
		return new_thread;
	}

	int run() {

switch(current_part) {
case 0: goto enter; 
case 1: goto sync1;
case 2: goto sync2;
case 3: goto sync3;
default: return -1;
}
enter:

    // func starts
    tid = threadId.x + blockId.x * gridDim.x;
    dim = gridDim.x * blockDim.x;
    seed = tid + (size_t) plane;
    
    while (true) {
        generated = control->n_generated;
        if (tid == 0) {
            if (generated >= settings->to_generate || 
                    control->n_errors >= settings->terminate_after) {
                control->terminated = 1;
            } else {
                control->has_error = 0;
                p_current = settings->circles + generated;
                if (randBool(&seed)) {
                    p_current->type = TRIAG_COLOR;
                    p_current->r = TRIAG_RADIUS;
                } else {
                    p_current->type = CIRCLE_COLOR;
                    p_current->r = CIRCLE_RADIUS;
                }
                p_current->x = randBetween(p_current->r, plane->actualWidth - p_current->r, &seed);
                p_current->y = randBetween(p_current->r, plane->actualHeight - p_current->r, &seed);
            }
        }
        current_part = 1; return 1;
sync1: __asm nop

        if (control->terminated) break;
        i = tid;
        p_current = settings->circles + generated;
        while (i < generated) {
            p_check = settings->circles + i;
            if (intersects(p_check, p_current)) {
                control->has_error = 1;
            }
            i += dim;
        }
        current_part = 2; return 1;
sync2: __asm nop

        if (tid == 0) {
            if (control->has_error) {
                control->n_errors++;
            } else {
                control->n_generated++;
                control->n_errors = 0;
            }
        }
        current_part = 3; return 1;
sync3: __asm nop

    }
    generated = control->n_generated;
    i = tid;
    actual_count->circles = 0;
    actual_count->triags = 0;
    while (i < generated) {
        if (CIRCLE_COLOR == settings->circles[i].type) {
            actual_count->circles += 1;
        } else {
            actual_count->triags += 1;
        }
        drawFigure(plane, settings->circles + i, &seed);
        i += dim;
    }
    return 0;

	}
};
