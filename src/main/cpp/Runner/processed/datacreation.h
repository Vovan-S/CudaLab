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


 size_t myrand(size_t* seed);
 bool randBool(size_t* seed);
 len_t randBetween(len_t lower_bound, len_t upper_bound, size_t* seed);

 bool intersects(Circle*, Circle*);

 int drawFigure(PlanePart*, Circle*, size_t* rand_seed);
 int drawLine(PlanePart* part, len_t x1, len_t y1, len_t x2, len_t y2);
 int drawCircle(PlanePart* part, len_t x, len_t y, len_t r);
 int drawPixel(PlanePart*, len_t x, len_t y, plane_t color);

 int create_data_gpu(PlanePart*, CreationSettingsStruct*, CreationControlStruct*, FigureCount*);
#include "_all_classes.h"