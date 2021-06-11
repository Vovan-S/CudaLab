#include "datacreation.h"

__device__ len_t randBetween(len_t lower_bound, len_t upper_bound, size_t* seed) {
    size_t N = 100000;
    len_t dx = (upper_bound - lower_bound) / N;
    return lower_bound + dx * (myrand(seed) % N);
}

__device__ bool intersects(Circle* c1, Circle* c2) {
    len_t dx = c1->x - c2->x;
    len_t dy = c1->y - c2->y;
    len_t dc2 = dx*dx + dy*dy;
    len_t r_diff = c1->r - c2->r;
    len_t r_sum = c1->r + c2->r;
    return !(dc2 > r_sum*r_sum + 10 || dc2 < r_diff*r_diff - 10);
}

__device__ int drawFigure(PlanePart* plane, Circle* circle, size_t* rand_seed) {
    if (CIRCLE_COLOR == circle->type) {
        drawCircle(plane, circle->x, circle->y, circle->r);
    } else {
        len_t x = randBetween(0, 1, rand_seed);
		len_t a = 1 - x * x;
        len_t y = a;
        for (int i = 0; i < 5; i++) y = (y + a / y) / 2;
        if (randBool(rand_seed)) y = -y;
        for (int i = 0; i < 3; i++) {
            len_t x1 = -0.5 * x + 0.866025 * y;
            len_t y1 = -0.866025 * x - 0.5 * y;
            drawLine(plane, circle->x + circle->r * x, 
                            circle->y + circle->r * y,
                            circle->x + circle->r * x1, 
                            circle->y + circle->r * y1);
            x = x1;
            y = y1;
        }
    }
    return 0;
}

__device__ int drawLine(PlanePart* plane, len_t x1, len_t y1, len_t x2, len_t y2) {
    len_t t, k, c;
    if (abs(x1 - x2) > abs(y1 - y2)) {
        if (x2 < x1) { t = x2; x2 = x1; x1 = t; t = y2; y2 = y1; y1 = t; }
        k = (y2 - y1) / (x2 - x1);
        for (size_t i = 0; i <= x2 - x1; i++) {
            c = y1 + k * i;
            drawPixel(plane, x1 + i, c, TRIAG_COLOR);
        }
    } else {
        if (y2 < y1) { t = x2; x2 = x1; x1 = t; t = y2; y2 = y1; y1 = t; }
        k = (x2 - x1) / (y2 - y1);
        for (size_t i = 0; i <= y2 - y1; i++) {
            c = x1 + k * i;
            drawPixel(plane, c, y1 + i, TRIAG_COLOR);
        }
    }
    return 0;
}

__device__ int drawPixel(PlanePart* plane, len_t x, len_t y, plane_t color) {
    plane->plane[(size_t) x + (size_t) y * plane->actualWidth] = color;
    return 0;
}

__device__ int drawEightPixels(PlanePart* plane, len_t cx, len_t cy, size_t dx, size_t dy) {
    drawPixel(plane, cx + dx, cy + dy, CIRCLE_COLOR);
    drawPixel(plane, cx + dx, cy - dy, CIRCLE_COLOR);
    drawPixel(plane, cx - dx, cy + dy, CIRCLE_COLOR);
    drawPixel(plane, cx - dx, cy - dy, CIRCLE_COLOR);
    
    drawPixel(plane, cx + dy, cy + dx, CIRCLE_COLOR);
    drawPixel(plane, cx + dy, cy - dx, CIRCLE_COLOR);
    drawPixel(plane, cx - dy, cy + dx, CIRCLE_COLOR);
    drawPixel(plane, cx - dy, cy - dx, CIRCLE_COLOR);
    return 0;
}

__device__ int drawCircle(PlanePart* plane, len_t x, len_t y, len_t r) {
    size_t cur_x = 0;
    size_t cur_y = (size_t) r;
    int d = (int) (3 - 2 * r);
    drawEightPixels(plane, x, y, cur_x, cur_y);
    for (; cur_x <= cur_y; ) {
        if (d < 0) {
            d = d + 4 * cur_x + 6;
            cur_x += 1;
        } else {
            d = d + 4 * (cur_x - cur_y) + 10;
            cur_x += 1;
            cur_y -= 1;
        }
        drawEightPixels(plane, x, y, cur_x, cur_y);
    }
    return 0;
}

__device__ size_t myrand(size_t* seed) {
    *seed = *seed * 22695477 + 1;
    return *seed;
}

__device__ bool randBool(size_t* seed) {
    return (myrand(seed) % 2) == 0;
}

__global__ int create_data_gpu(PlanePart* plane, 
                               CreationSettingsStruct* settings, 
                               CreationControlStruct* control,
                               FigureCount* actual_count) {
    // local vars 
    size_t tid, generated, i, dim, seed;
    Circle *p_current;
    Circle *p_check;
    
    // func starts
    tid = threadId.x + blockId.x * blockDim.x;
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
        __syncthreads();
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
        __syncthreads();
        if (tid == 0) {
            if (control->has_error) {
                control->n_errors++;
            } else {
                control->n_generated++;
                control->n_errors = 0;
            }
        }
        __syncthreads();
    }
    generated = control->n_generated;
    if (tid == 0) {
        actual_count->circles = 0;
        actual_count->triags = 0;
        for (i = 0; i < generated; i++) {
            if (CIRCLE_COLOR == settings->circles[i].type) {
                actual_count->circles += 1;
            } else {
                actual_count->triags += 1;
            }
        }
    }
    i = tid;
    while (i < generated) {
        drawFigure(plane, settings->circles + i, &seed);
        i += dim;
    }
    return 0;
}

err_t create_data(PlanePart* plane, FigureCount desired, FigureCount* actual) {
    int M = desired.circles + desired.triags;
    
    Circle* circles;
    cudaMalloc((void **)&circles, M * sizeof(Circle));
    
    CreationSettingsStruct* dev_css;
    cudaMalloc((void **)&dev_css, sizeof(CreationSettingsStruct));
    CreationSettingsStruct hst_css;
    hst_css.circles = circles;
    hst_css.to_generate = M;
    hst_css.terminate_after = 10;
    cudaMemcpy(dev_css, &hst_css, sizeof(CreationSettingsStruct), cudaMemcpyHostToDevice);
    
    CreationControlStruct* ccs;
    cudaMalloc((void **)&ccs, sizeof(CreationControlStruct));
    cudaMemset(ccs, 0, sizeof(CreationControlStruct));
    
    FigureCount* fc; 
    cudaMalloc((void **)&fc, sizeof(FigureCount));
    cudaMemset(fc, 0, sizeof(FigureCount));
    
    create_data_gpu<<<1, MIN(M, THREADS_CREATION)>>>(plane, dev_css, ccs, fc);
    
    cudaMemcpy(actual, fc, sizeof(FigureCount), cudaMemcpyDeviceToHost);
    
    cudaFree(fc);
    cudaFree(dev_css);
    cudaFree(circles);
    cudaFree(ccs);
    
    return 0;
}
