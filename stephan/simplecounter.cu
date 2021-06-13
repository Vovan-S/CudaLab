#include "SimpleCounter.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>



// Функция Вовы <-- убери это, если будешь вставлять в отчет :)
err_t count_figures(PlanePart* plane, FigureCount* result, CudaConfig cuda_cfg) {
    
    // Общее количество всех-всех
    int M = cuda_cfg.nBlocks * cuda_cfg.nThreads;
    
    // Моя структура данных для полигона (см. header.cuh)
    // она лежит на device, поэтому копирую
    PlanePart hst_plane;
    cudaMemcpy(&hst_plane, plane, sizeof(PlanePart), cudaMemcpyDeviceToHost);
    
    // Массивы для подсчета
    int *dev_c, *dev_t;
    cudaMalloc(&dev_c, sizeof(int) * M);
    cudaMalloc(&dev_t, sizeof(int) * M);
    cudaMemset(dev_c, 0, sizeof(int) * M);
    cudaMemset(dev_t, 0, sizeof(int) * M);
    
    // TODO: допили логику для iter, d, dones плз
    runBlocks<<<cuda_cfg.nBlocks, cuda_cfg.nThreads>>>( ??? iter ???,
                                                        hst_plane.actualHeight,
                                                        hst_plane.plane,
                                                        ???  d  ???,
                                                        dev_c, 
                                                        dev_t,
                                                        ??? dones ???);
    // Время для редукции!
    int red_threads = 1;
    while (red_threads * 2 < cuda_cfg.nThreads) red_threads *= 2;
    reduction<<<1, red_threads>>>(dev_c, dev_t, M);
    
    
    cudaMemcpy(&(result->circles), dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&(result->triags), dev_t, sizeof(int), cudaMemcpyDeviceToHost);
    
    
    cudaFree(dev_c);
    cudaFree(dev_t);
    return 0;
}

// Простая редукция на один блок
__global__ void reduction(int* circles, int* triangs, int size) {
    if (blockIdx.x != 0) return;
    int tid = threadIdx.x;
    int dim = blockDim.x;
    for (int i = tid + dim; i < size; i += dim) {
        circles[tid] += circles[i];
        triangs[tid] += triangs[i];
    }
    __synchthreads;
    int i = dim / 2;
    while (i != 0) {
        if (tid + i < dim) {
            circles[tid] += circles[tid + i];
            triangs[tid] += triangs[tid + i];
        }
        __synchthreads;
        i /= 2;
    }
}

__global__ void runBlocks(int iter, int polySize, int* polygon, Point d, int* circles, int* triangs,
	thrust::device_vector<int>* dones)
{
	// iter -- номер шага работы алгоритма (за iter шагов построено слияние для 2^(iter-1) * d клеток)
	// polySize -- размер стороны полигона
	// polygon -- указатель на начало полигона
	// circles -- указатель на счетчик кругов
	// triangs -- указатель на счетчик треугольников
	// dones -- таблица фигур, распознанных каждой нитью 
	
	Point xy = getXY();
	Point start = Point(xy.x * d.x, xy.y * d.y);

	thrust::device_vector<int>& done = *getDonesPoint(dones);

	// выполняем редукцию
	// iter -> управляем нитью, отвечающей за квадрат iter * iter сегментов
	Point end = start + d * iter;

	Point xy = getXY();
	if (xy.x % iter == 0 && xy.y % iter == 0)
	{
		if (iter == 1)
			// проходимся по всему полигону
			for (int x = start.x; x < end.x; x++)
				for (int y = start.y; y < end.y; y++)
				{
					int idx = getByPoint(Point(x, y), polySize);
					if (
						(polygon[idx] == CIRCLE_COLOR || polygon[idx] == TRIANG_COLOR) &&
							!contains(done, idx)
						)
						if (polygon[idx] == CIRCLE_COLOR)
							*circles += make_cycle(idx, polySize, polygon, start, end, done);
						else
							*triangs += make_cycle(idx, polySize, polygon, start, end, done);
				}
		else
			// проходимся только по "центральному перекрестью"
		{
			uniteDones(dones, iter); // сначала добавили к себе всё, что знают соседи
			int cx = start.x + (end.x - start.x) / 2,
				cy = start.y + (end.y - start.y) / 2;
			for (int x = start.x; x < end.x; x++)
			{
				int idx = getByPoint(Point(x, cy), polySize);
				if (
					(polygon[idx] == CIRCLE_COLOR || polygon[idx] == TRIANG_COLOR) &&
					!contains(done, idx)
					)
					if (polygon[idx] == CIRCLE_COLOR)
						*circles += make_cycle(idx, polySize, polygon, start, end, done);
					else
						*triangs += make_cycle(idx, polySize, polygon, start, end, done);
			}

			for (int y = start.y; y < end.y; y++)
			{
				int idx = getByPoint(Point(cx, y), polySize);
				if (
					(polygon[idx] == CIRCLE_COLOR || polygon[idx] == TRIANG_COLOR) &&
					!contains(done, idx)
					)
					if (polygon[idx] == CIRCLE_COLOR)
						*circles += make_cycle(idx, polySize, polygon, start, end, done);
					else
						*triangs += make_cycle(idx, polySize, polygon, start, end, done);
			}
		}
	}
}

__device__ int make_cycle(int idx, int polySize, int* polygon, Point start, Point end,
	thrust::device_vector<int>& done)
{
	int next = done.size();
	thrust::device_vector<int> todo;
	Point t = getByCoords(idx, polySize);
	todo.push_back(idx);

	while (todo.size())
	{
		int next = todo.front();
		todo.erase(todo.begin());

		done.push_back(next);
		addNeigh(todo, getByCoords(next, polySize), start, end, done, polySize, polygon);
	}

	auto start_neighs = neighs(getByCoords(done.back(), polySize), start, end, polySize, polygon);
	if (contains(start_neighs, idx))
		return 1;
	else
	{
		// нужно удалить из done все добавленные точки
		if (done.size() > next)
			done.erase(done.begin() + next, done.end());
		return 0;
	}
}

__device__ thrust::device_vector<int> neighs(Point pos, Point& start, Point& stop, int polySize, int* polygon)
{
	thrust::device_vector<int> res;
	int pos_idx = getByPoint(pos, polySize);

	for (int x = pos.x - 1; x <= pos.x + 1; x++)
		for (int y = pos.y - 1; y <= pos.y + 1; y++)
		{
			Point xy(x, y);
			if (xy == pos)
				continue;
			int idx = getByPoint(xy, polySize);
			if (polygon[idx] == polygon[pos_idx] && !outBorder(pos, start, stop))
				res.push_back(idx);
		}
	return res;
}

__device__ void uniteDones(thrust::device_vector<int>* dones, int iter)
{
	iter /= 2; // iter -- расстояние до вектора с вершинами соседа
	auto me = getDonesPoint(dones),
		right = getDonesPoint(dones, { iter, 0 }),
		bottom = getDonesPoint(dones, { 0, iter }),
		rightbottom = getDonesPoint(dones, { iter, iter });

	me->insert(me->end(), right->begin(), right->end());
	me->insert(me->end(), bottom->begin(), bottom->end());
	me->insert(me->end(), rightbottom->begin(), rightbottom->end());

	right->clear();
	bottom->clear();
	rightbottom->clear();
}

__device__ bool onBorder(Point t, Point& start, Point& end)
{
	return t.x == start.x || t.y == start.y || t.x == end.x || t.y == end.y;
}

__device__ bool outBorder(Point t, Point& start, Point& end)
{
	return t.x < start.x || t.y < start.y || t.x > end.x || t.y > end.y;
}

__device__ bool contains(thrust::device_vector<int> &v, int p)
{
	for (auto point : v)
		if (point == p)
			return true;
	return false;
}

__device__ void addNeigh(thrust::device_vector<int>& stack, Point pos, Point& start, Point& stop,
	thrust::device_vector<int>& done, int polySize, int* polygon)
{
	// Добавить в стэк еще не обойденных соседей
	int pos_idx = getByPoint(pos, polySize);

	for (int x = pos.x - 1; x <= pos.x + 1; x++)
		for (int y = pos.y - 1; y <= pos.y + 1; y++)
		{
			Point xy(x, y);
			if (xy == pos)
				continue;
			int idx = getByPoint(xy, polySize);
			if (polygon[idx] == polygon[pos_idx] && !contains(done, idx) && !outBorder(pos, start, stop))
				stack.push_back(idx);
		}
}

__device__ int _abs(int x)
{
	return x > 0 ? x : -x;
}

// ====================================================================================================

__device__ Point getXY()
{
	return Point(threadIdx.x, blockIdx.x);
}

__device__ thrust::device_vector<int>* getDonesPoint(thrust::device_vector<int>* dones, Point d)
{
	return dones + (threadIdx.x + d.x) + (blockIdx.x + d.y) * gridDim.x;
}

__device__ int getByPoint(Point p, int polySize)
{
	return p.y * polySize + p.x;
}

__device__ Point getByCoords(int idx, int polySize)
{
	return Point(idx % polySize, idx / polySize);
}   