#pragma once
#include "MainHeader.h"

#define REDUCTION_THREADS 256

typedef struct {
    size_t k;    
    size_t len;  
} CountAlgConfig;

typedef struct {
    char direction; 
    size_t rang;    
    size_t cx;      
    size_t cy;      
    size_t start;   
    size_t stop;    
} LineConfig;


typedef struct {
    size_t x[10];
    size_t y[10];
    char size;
    char start;
} SmallQuery;

 bool isConnected(size_t x, size_t y, PlanePart * plane, plane_t color);
 plane_t pixel(PlanePart*, size_t x, size_t y);

 LineConfig map(size_t tid, CountAlgConfig*);

 void sq_init(SmallQuery*);
 int sq_push(SmallQuery*, size_t x, size_t y);
 int sq_pop(SmallQuery*, size_t* x, size_t* y);
 int sq_contains(SmallQuery*, size_t x, size_t y);

 int traverse(PlanePart*, SmallQuery*, size_t len, size_t* x, size_t* y, plane_t color);
 int should_count(size_t x, size_t y, LineConfig*, size_t len);


 int count_figures_gpu(PlanePart*, CountStruct*, CountAlgConfig);
 int reduction(CountStruct*, FigureCount*, size_t elems);
#include "_all_classes.h"