#include "runner.cuh"


int runPerfomanceTests(TestRunInstance* ptri, size_t n_tests, std::ostream& log) {
    err_t er;
    FigureCount expected, got;
    cudaEvent_t start, stop;
    size_t nSuccessful = 0;
    size_t nCrashed = 0;
    size_t nErrors = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    log << "Test runs. Preparing to run " << n_tests << " tests.\n";
    for (size_t i = 0; i < n_tests; i++) {
        log << "\nRunning test #" << (i + 1) << std::endl;
        TestRunInstance* t = ptri + i;
        log << "Blocks:\t\t" << t->cfg.nBlocks << std::endl;
        log << "Threads:\t" << t->cfg.nThreads << std::endl;
        //ASSERT_TRI(t);
        if (t->regenerate) {
            TRY(create_data(t->plane, t->figures, &expected, i), "data creation");
        }
        cudaEventRecord(start, 0);
        TRY(count_figures(t->plane, &got, t->cfg), "counting figures");
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&(t->ms), start, stop);
        if (expected.circles == got.circles && expected.triags == got.triags) {
            t->successful = true;
            log << "Ran successful, time = " << t->ms << std::endl;
            nSuccessful++;
            continue;
        } else {
            log << "Counting error. Expected: (c=" << expected.circles << ", t=" << 
                expected.triags << "), got: (c=" << got.circles << ", t=" << got.triags <<
                ").\n";
            log << "Time = " << t->ms << "\n";
            nErrors++;
            t->successful = false;
            continue;
        }
error:
        nCrashed++;
        t->successful = false;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);    
    log << "\nRan " << n_tests << " tests.\nSuccessful:\t" << nSuccessful <<
        "\nCrashed:\t" << nCrashed << "\nErrors:\t\t" << nErrors << std::endl;
    return 0;
}