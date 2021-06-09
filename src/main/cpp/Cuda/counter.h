#pragma once
#include "MainHeader.h"

#define REDUCTION_THREADS 256

typedef struct {
    size_t k;    // количество порядков зон
    size_t len;  // длина границы
} CountAlgConfig;

typedef struct {
    char direction; // 0 - горизонатальная граница, 1 - вертикальная
    size_t rang;    // порядок зоны
    size_t cx;      //  \ центр  
    size_t cy;      //  / зоны
    size_t start;   // значение свободной координаты в начале границы
    size_t stop;    // значение свободной координаты в конце границы
} LineConfig;


typedef struct {
    size_t x[10];
    size_t y[10];
    char size;
    char start;
} SmallQuery;

__device__ bool isConnected(size_t x, size_t y, PlanePart * plane, plane_t color);
__device__ plane_t pixel(PlanePart*, size_t x, size_t y);

__device__ LineConfig map(size_t tid, CountAlgConfig*);

__device__ void sq_init(SmallQuery*);
__device__ int sq_push(SmallQuery*, size_t x, size_t y);
__device__ int sq_pop(SmallQuery*, size_t* x, size_t* y);
__device__ int sq_contains(SmallQuery*, size_t x, size_t y);

__device__ int traverse(PlanePart*, SmallQuery*, size_t len, size_t* x, size_t* y, plane_t color);
__device__ int should_count(size_t x, size_t y, LineConfig*, size_t len);


__global__ int count_figures_gpu(PlanePart*, CountStruct*, CountAlgConfig);
__global__ int reduction(CountStruct*, FigureCount*, size_t elems);