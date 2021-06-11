#pragma once
#include "cuda_runtime.h"

//===================================================================================//
//=================================MainHeader.h======================================//
//===================================================================================//


#include <inttypes.h>


// Тип для точек плоскости
typedef uint8_t plane_t;
// Тип для ошибок
typedef int err_t;
// Тип для любых подсчетов -- size_t

#define TRIAG_COLOR 2
#define CIRCLE_COLOR 3

#define ERRCODE -1


// Структура для полигона
typedef struct { 
	plane_t* plane;     // указатель на начало обрабатываемой части
	size_t width;       // ширина обрабатываемой части
	size_t height;      // высота обрабатываемой части
	size_t actualWidth; // ширина всего полигона
	size_t actualHeight;// высота всего полигона
} PlanePart;

// Структура для подсчетов
typedef struct {
	size_t* triag;  // счетчик для треугольников
	size_t* circle; // счетчик для кругов
} CountStruct;

// Количество фигур
typedef struct {
    size_t circles;
    size_t triags;
} FigureCount;

typedef struct {
    size_t nBlocks;
    size_t nThreads;
} CudaConfig;

// Этапы работы программы:
// 1. Генерация входных данных
//    Получаем на вход полигон (на device!), желаемое количество фигур и 
//    указатель на итоговое количество фигур (память на host!).
err_t create_data(PlanePart* plane, FigureCount desired, FigureCount* actual);

// 2. Решение задачи
//    Полигон выделен на device, result на хосте
err_t count_figures(PlanePart* plane, FigureCount* result, CudaConfig cfg);


//===================================================================================//
//==================================counter.h========================================//
//===================================================================================//


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



//===================================================================================//
//================================datacreation.h=====================================//
//===================================================================================//


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


//===================================================================================//
//===================================runner.h========================================//
//===================================================================================//


#include <ostream>

#define MAX_THREADS 256
#define MAX_BLOCKS (size_t) 1 << 16

#define TRY(x, y) if ((x) != 0) { log << "Error with " << y << std::endl; goto error; }
#define ASSERT_PLANE(x) if ((x) == NULL || (x)->plane == NULL || (x)->plane[0] > 3) { log << "Plane assertion fails.\n"; goto error;}
#define ASSERT_CUDACFG(x) if ((x).nBlocks > MAX_BLOCKS || (x).nThreads > MAX_THREADS) {log << "Cuda config assertion fails.\n"; goto error;}
#define ASSERT_TRI(x) ASSERT_PLANE((x)->plane) ASSERT_CUDACFG((x)->cfg)

typedef struct {
    PlanePart* plane;
    CudaConfig cfg;
    FigureCount figures;
    bool regenerate;
    bool successful;
    float ms;
} TestRunInstance;

int runPerfomanceTests(TestRunInstance*, size_t n_tests, std::ostream& log);
