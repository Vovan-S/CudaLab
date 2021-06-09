#include "Alg.h"

#include <fstream>

int main() {
    const int N = 3000;
    const int M = 50;
    plane_t* plane = new plane_t[N * N];
    memset(plane, 0, N * N * sizeof(plane_t));
    PlanePart ps;
    ps.plane = plane;
    ps.actualWidth = N;
    ps.actualHeight = N;
    Circle* circles = new Circle[M];
    CreationSettingsStruct css;
    css.circles = circles;
    css.to_generate = M;
    css.terminate_after = 10;
    CreationControlStruct ccs;
    memset(&ccs, 0, sizeof(ccs));
    FigureCount fc;
    memset(&fc, 0, sizeof(fc));
    create_data_gpu<<<1, M>>>(&ps, &css, &ccs, &fc);
    std::ofstream fout;
    fout.open("file.bin", std::ios::binary | std::ios::out);
    fout.write((char*)plane, N * N);
    fout.close();
}