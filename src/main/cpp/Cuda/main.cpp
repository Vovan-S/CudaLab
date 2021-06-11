#include "runner.h"

#include <iostream>

int main() {
    PlanePart* dev_plane;
    PlanePart  hst_plane;
    int N = 3000;
    
    cudaMalloc((void **)&(hst_plane.plane), sizeof(plane_t) * N * N);
    cudaMemset(hst_plane.plane, 0, sizeof(plane_t) * N * N);
    hst_plane.height = hst_plane.width = N;
    hst_plane.actualHeight = hst_plane.actualWidth = N;
    
    cudaMalloc((void **)&dev_plane, sizeof(PlanePart));
    cudaMemcpy(dev_plane, &hst_plane, sizeof(PlanePart), cudaMemcpyHostToDevice);
    
    TestRunInstance tests[1];
    tests[0].plane = dev_plane;
    tests[0].cfg = {256, 256};
    tests[0].figures = {100, 100};
    tests[0].regenerate = true;
    
    runPerfomanceTests(tests, 1, std::cout);
    
    cudaFree(hst_plane.plane);
    cudaFree(dev_plane);
    return 0;
}