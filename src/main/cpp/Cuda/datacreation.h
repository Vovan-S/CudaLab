#pragma once
#include "MainHeader.h"


#define THREADS_CREATION 50

#define MIN(x, y) (x > y)? y : x

#define CIRCLE_RADIUS 100.0
#define TRIAG_RADIUS 57.773502 

typedef float len_t;

typedef struct {
    len_t x;
    len_t y;
    len_t r;
    plane_t type;
} Circle;

typedef struct {
    Circle* circles;
    size_t to_generate;
    size_t terminate_after;
} CreationSettingsStruct;

typedef struct {
    bool terminated;
    size_t n_generated;
    size_t n_errors;
    bool has_error;
} CreationControlStruct;


__device__ size_t myrand(size_t* seed);
__device__ bool randBool(size_t* seed);
__device__ len_t randBetween(len_t lower_bound, len_t upper_bound, size_t* seed);

__device__ bool intersects(Circle*, Circle*);

__device__ int drawFigure(PlanePart*, Circle*, size_t* rand_seed);
__device__ int drawLine(PlanePart* part, len_t x1, len_t y1, len_t x2, len_t y2);
__device__ int drawCircle(PlanePart* part, len_t x, len_t y, len_t r);
__device__ int drawPixel(PlanePart*, len_t x, len_t y, plane_t color);

__global__ int create_data_gpu(PlanePart*, CreationSettingsStruct*, CreationControlStruct*, FigureCount*);