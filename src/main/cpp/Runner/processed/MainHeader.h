#pragma once

#include <inttypes.h>



typedef uint8_t plane_t;

typedef int err_t;


#define TRIAG_COLOR 2
#define CIRCLE_COLOR 3

#define ERRCODE -1



typedef struct { 
	plane_t* plane;     
	size_t width;       
	size_t height;      
	size_t actualWidth; 
	size_t actualHeight;
} PlanePart;


typedef struct {
	size_t* triag;  
	size_t* circle; 
} CountStruct;


typedef struct {
    size_t circles;
    size_t triags;
} FigureCount;

typedef struct {
    size_t nBlocks;
    size_t nThreads;
} CudaConfig;





err_t create_data(PlanePart* plane, FigureCount desired, FigureCount* actual);



err_t count_figures(PlanePart* plane, FigureCount* result, CudaConfig cfg);
