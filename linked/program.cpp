#include "program.h"

int main() {
    PlanePart* dev_plane;
    PlanePart  hst_plane;
    int N = 3000;
    
    cudaMalloc((void **)&(hst_plane.plane), sizeof(plane_t) * N * N);
    cudaMemset(hst_plane.plane, 0, sizeof(plane_t) * N * N);
    hst_plane.height = hst_plane.width = N;
    hst_plane.actualHeight = hst_plane.actualWidth = N;
    
    cudaMalloc((void **)&dev_plane, sizeof(PlanePart));
    cudaMemcpy(dev_plane, &hst_plane, sizeof(PlanePart), cudaMemcpyHostToDevice);
    
    TestRunInstance tests[1];
    tests[0].plane = dev_plane;
    tests[0].cfg = {256, 256};
    tests[0].figures = {100, 100};
    tests[0].regenerate = true;
    
    runPerfomanceTests(tests, 1, std::cout);
    
    cudaFree(hst_plane.plane);
    cudaFree(dev_plane);
    return 0;
}

err_t count_figures(PlanePart* plane, FigureCount* result, CudaConfig cuda_cfg) {
    size_t N = cuda_cfg.nBlocks * cuda_cfg.nThreads;
    
    PlanePart hst_plane; 
    cudaMemcpy(&hst_plane, plane, sizeof(PlanePart), cudaMemcpyDeviceToHost);
    
    CountStruct* dev_count;
    CountStruct hst_count;
    
    cudaMalloc((void **)&(hst_count.triag), N * sizeof(size_t));
    cudaMalloc((void **)&(hst_count.circle), N * sizeof(size_t));
    cudaMalloc((void **)&dev_count, sizeof(CountStruct));
    
    cudaMemcpy(dev_count, &hst_count, sizeof(CountStruct), cudaMemcpyHostToDevice);
    
    CountAlgConfig cfg; 
    for (int i = sizeof(size_t) * 8 - 1; i >= 0; i--) {
        if (hst_plane.actualHeight & (1 << i)) {
            cfg.k = i - 5;
            cfg.len = hst_plane.actualHeight / (1 << cfg.k) + 1;
            break;
        }
    }
    
    count_figures_gpu<<<cuda_cfg.nBlocks, cuda_cfg.nThreads>>>(plane, dev_count, cfg);
    
    FigureCount* dev_fc;
    
    cudaMalloc((void **)&dev_fc, sizeof(FigureCount));
   
    reduction<<<1, cuda_cfg.nThreads>>>(dev_count, dev_fc, N);
    
    cudaMemcpy(result, dev_fc, sizeof(FigureCount), cudaMemcpyDeviceToHost);
   
    cudaFree(dev_fc);
    cudaFree(hst_count.circle);
    cudaFree(hst_count.triag);
    cudaFree(dev_count);
    
    return 0;
}

__device__ plane_t pixel(PlanePart* plane, size_t x, size_t y) {
    return plane->plane[x + y * plane->actualWidth];
}

__device__ LineConfig map(size_t tid, CountAlgConfig* cfg) {
    LineConfig res;
    size_t k = cfg->k;
    size_t n = 1 << k;
    res.direction = tid / (n * (n - 1));
    tid = tid % (n * (n - 1));
    size_t slice = tid / n;
    size_t part = tid % n;
    size_t r = 0;
    while (!((slice + 1) & (1 << r))) r++;
    res.rang = r;
    size_t zone_size = 1 << (r + 1);
    res.cx = ((part / zone_size) * zone_size + zone_size / 2) * cfg->len;
    res.cy = (slice + 1) * cfg->len;
    if (res.direction) {
        size_t t = res.cx;
        res.cx = res.cy;
        res.cy = t;
    }
    res.start = cfg->len * part;
    res.stop = res.start + cfg->len;
    return res;
}

__device__ void sq_init(SmallQuery* sq) {
    sq->size = sq->start = 0;
}


__device__ int sq_push(SmallQuery* sq, size_t x, size_t y) {
    sq->x[(sq->start + sq->size) % 10] = x;
    sq->y[(sq->start + sq->size) % 10] = y;
    sq->size++;
    if (sq->size > 10) return 1;
    else return 0;
}


__device__ int sq_pop(SmallQuery* sq, size_t* x, size_t* y) {
    if (sq->size == 0) return 1;
    *x = sq->x[sq->start];
    *y = sq->y[sq->start];
    sq->start = (sq->start + 1) % 10;
    sq->size--;
    return 0;
}

__device__ int sq_contains(SmallQuery* sq, size_t x, size_t y) {
    for (int i = 0; i < 10; i++) {
        if (sq->x[i] == x) {
            if (sq->y[i] == y) return 1;
        }
    }
    return 0;
}

#define insert_correct(x, y) if (x >= 0 && x < plane->actualWidth && y >= 0 && y < plane->height && !sq_contains(sq, x, y) && pixel(plane, x, y) == color) sq_push(sq, x, y);

__device__ int traverse(PlanePart* plane, SmallQuery* sq, size_t len, size_t* cross_x, size_t* cross_y, plane_t color) {
    size_t x, y;
    size_t watchdog = 0;
    while(!sq_pop(sq, &x, &y)) {
        if (watchdog++ > 1000) return 0;
        insert_correct(x - 1, y - 1);
        insert_correct(x - 1, y);
        insert_correct(x - 1, y + 1);
        insert_correct(x, y - 1);
        insert_correct(x, y + 1);
        insert_correct(x + 1, y - 1);
        insert_correct(x + 1, y);
        insert_correct(x + 1, y + 1);
        if (x % len == 0 || y % len == 0) {
            *cross_x = x;
            *cross_y = y;
            return 1;
        }
    }
    return 0;
}

__device__ int should_count(size_t x, size_t y, LineConfig* cfg, size_t len) {
     int dcx = x - cfg->cx;
     int dcy = y - cfg->cy;
     if ((dcx % ((int)len << cfg->rang) == 0 && dcx != 0) ||
         (dcy % ((int)len << cfg->rang) == 0 && dcy != 0)) {
         return 0;
     }
     if (cfg->direction == 1) {
         if (y < cfg->stop) return 1;
         else return 0;
     }
     else {
         if (y < cfg->cy) return 1;
         else if (y > cfg->cy) return 0;
         else if (x < cfg->stop) return 1;
         else return 0;
     }
}

__global__ int count_figures_gpu(PlanePart* plane, CountStruct* count, CountAlgConfig alg_cfg) {
    // local vars
    size_t tid, x, y, dim, i, x1, y1;
    plane_t color;
    char visited[100];
    LineConfig cfg;
    SmallQuery sq;
    char flag;
    
    // func starts
    tid = threadId.x + blockId.x * blockDim.x;
    dim = gridDim.x * blockDim.x;
    
    count->triag[tid] = 0;
    count->circle[tid] = 0;

    
    while(tid < 2 * (1 << alg_cfg.k) * ((1 << alg_cfg.k) - 1)) {
        cudaMemset(visited, 0, sizeof(visited));
        cfg = map(tid, &alg_cfg);
        for (i = 0; i < alg_cfg.len; i++) {
            if (0 == cfg.direction) { // горизонтальная граница
                x = cfg.start + i;
                y = cfg.cy;
                if (x >= plane->actualWidth) break;
            }
            else {
                x = cfg.cx;
                y = cfg.start + i;
                if (y >= plane->actualHeight) break;

            }
            if (!visited[i]) {
                color = pixel(plane, x, y);
                if (TRIAG_COLOR == color || CIRCLE_COLOR == color) {
                    sq_init(&sq);
                    sq_push(&sq, x, y);
                    visited[i] = 1;
                    flag = 1;
                    while (traverse(plane, &sq, alg_cfg.len << cfg.rang, &x1, &y1, color)) {
						if (x1 == x && y1 == y) continue;
						if (cfg.direction == 0 && y1 == y && x1 < cfg.stop) {
							visited[x1 - cfg.start] = 1;
							continue;
						}
						if (cfg.direction == 1 && x1 == x && y1 < cfg.stop) {
							visited[y1 - cfg.start] = 1;
							continue;
						}
                        if (!should_count(x1, y1, &cfg, alg_cfg.len)) {
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

__global__ int reduction(CountStruct* count, FigureCount* fc, size_t elems) {
    // local vars
    size_t tid, dim, i, j;
    
    // func starts
    if (blockId.x != 0) return 0;
    tid = threadId.x;
    dim = blockDim.x;
    
    for (i = 1; tid + i * dim < elems; i++) {
        count->circle[tid] += count->circle[tid + i * dim];
        count->triag[tid] += count->triag[tid + i * dim];
    }
    __syncthreads();
    i = dim / 2;
    while (i != 0) {
        if (tid < i && tid + i < elems) {
            count->circle[tid] += count->circle[tid + i];
            count->triag[tid] += count->triag[tid + i];
        }
        __syncthreads();
        i /= 2;
    }
    fc->circles = count->circle[0];
    fc->triags = count->triag[0];
    return 0;
}

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

int runPerfomanceTests(TestRunInstance* ptri, size_t n_tests, std::ostream& log) {
    err_t er;
    FigureCount expected, got;
    cudaEvent_t start, stop;
    size_t nSuccessful = 0;
    size_t nCrashed = 0;
    size_t nErrors = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    for (size_t i = 0; i < n_tests; i++) {
        log << "Running test #" << (i + 1) << std::endl;
        TestRunInstance* t = ptri + i;
        ASSERT_TRI(t);
        if (t->regenerate) {
            TRY(create_data(t->plane, t->figures, &expected), "data creation");
        } else {
            expected = t->figures;
        }
        cudaEventRecord(start, 0);
        TRY(count_figures(t->plane, &got, t->cfg), "counting figures");
        cudaEventRecord(stop, 0);
        cudaEventSyncronize(stop);
        if (expected.circles == got.circles && expected.triags == got.triags) {
            cudaEventElapsedTime(&(t->ms), start, stop);
            t->successful = true;
            log << "Ran successful, time = " << t->ms << std::endl;
            nSuccessful++;
            continue;
        } else {
            log << "Counting error. Expected: (c=" << expected.circles << ", t=" << 
                expected.triags << "), got: (c=" << got.circles << ", t=" << got.triags <<
                ").\n";
            nErrors++;
            t->successful = false;
            continue;
        }
error:
        nCrashed++;
        t->successful = false;
    }
    log << "Ran " << n_tests << " tests.\nSuccessful:\t" << nSuccessful <<
        "\nCrashed:\t" << nCrashed << "\nErrors:\t\t" << nErrors << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
