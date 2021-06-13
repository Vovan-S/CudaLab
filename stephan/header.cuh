#pragma once
#include "cuda_runtime.h"
#include <iostream>
#include <inttypes.h>

// Тип для точек плоскости
//  UPD: у меня было байтиками, но для тебя сделал интами
typedef int plane_t;
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
err_t create_data(PlanePart* plane, FigureCount desired, FigureCount* actual, size_t seed = 0);

// 2. Решение задачи
//    Полигон выделен на device, result на хосте
err_t count_figures(PlanePart* plane, FigureCount* result, CudaConfig cfg);
