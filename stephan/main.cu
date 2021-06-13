#include "runner.cuh"

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
    
    size_t threads[] = {32, 64, 128, 256};
    size_t blocks[]  = {16, 32, 64, 128, 256, 512};
    
    int nTests = sizeof(threads) / sizeof(threads[0]) * 
                    sizeof(blocks) / sizeof(blocks[0]);
    
    TestRunInstance* tests = new TestRunInstance[nTests];
    for (int i = 0; i < sizeof(threads) / sizeof(threads[0]); i++) 
        for (int j = 0; j < sizeof(blocks) / sizeof(blocks[0]); j++) {
            int k = j + sizeof(blocks) / sizeof(blocks[0]) * i;
            tests[k].plane = dev_plane;
            tests[k].cfg.nThreads = threads[i];
            tests[k].cfg.nBlocks = blocks[j];
            tests[k].regenerate = false;
        }    
    tests[0].figures.circles = 100;
    tests[0].figures.triags = 100;
    tests[0].regenerate = true;
    
    runPerfomanceTests(tests, nTests, std::cout);
        
    
    delete[] tests;
    cudaFree(hst_plane.plane);
    cudaFree(dev_plane);
    return 0;
}
