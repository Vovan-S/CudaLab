#include "counter.h"

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
