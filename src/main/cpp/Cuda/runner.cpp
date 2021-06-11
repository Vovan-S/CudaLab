#include "runner.h"

int runPerfomanceTests(TestRunInstance* ptri, size_t n_tests, std::ostream& log) {
    err_t er;
    FigureCount expected, got;
    cudaEvent_t start, stop;
    size_t nSuccessful = 0;
    size_t nCrashed = 0;
    size_t nErrors = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;
    for (size_t i = 0; i < n_tests; i++) {
        log << "Running test #" << (i + 1) << std::endl;
        TestRunInstance* t = ptri + i;
        ASSERT_TRI(t);
        if (t->regenerate) {
            TRY(create_data(t->plane, t->figures, &expected), "data creation");
        } else {
            expected = t->figures;
        }
        cudaEventRecord(start, 0);
        TRY(count_figures(t->plane, &got, t->cfg), "counting figures");
        cudaEventRecord(stop, 0);
        cudaEventSyncronize(stop);
        if (expected.circles == got.circles && expected.triags == got.triags) {
            cudaEventElapsedTime(&(t->ms), start, stop);
            t->successful = true;
            log << "Ran successful, time = " << t->ms << std::endl;
            nSuccessful++;
            continue;
        } else {
            log << "Counting error. Expected: (c=" << expected.circles << ", t=" << 
                expected.triags << "), got: (c=" << got.circles << ", t=" << got.triags <<
                ").\n";
            nErrors++;
            t->successful = false;
            continue;
        }
error:
        nCrashed++;
        t->successful = false;
    }
    log << "Ran " << n_tests << " tests.\nSuccessful:\t" << nSuccessful <<
        "\nCrashed:\t" << nCrashed << "\nErrors:\t\t" << nErrors << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
