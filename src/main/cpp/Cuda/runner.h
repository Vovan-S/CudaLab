#pragma once
#include "MainHeader.h"

#include <ostream>

#define MAX_THREADS 256
#define MAX_BLOCKS (size_t) 1 << 48

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