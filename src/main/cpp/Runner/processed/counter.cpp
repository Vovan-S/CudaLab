#include "counter.h"

#include "_all_classes.h"
err_t count_figures(PlanePart* plane, FigureCount* result, CudaConfig cfg) {
    return 0;
}

 plane_t pixel(PlanePart* plane, size_t x, size_t y) {
    return plane->plane[x + y * plane->actualWidth];
}

 LineConfig map(size_t tid, CountAlgConfig* cfg) {
    LineConfig res;
    size_t k = cfg->k;
    res.direction = tid / (k * (k - 1));
    tid = tid % (k * (k - 1));
    size_t slice = tid / k;
    size_t part = tid % k;
    size_t r = 0;
    while (!((slice + 1) & (1 << r))) r++;
    res.rang = r;
    size_t zone_size = 1 << (r + 1);
    res.cx = ((part / zone_size) * zone_size + zone_size / 2) * cfg->len;
    res.cy = (slice + 1) * cfg->len;
    if (res.direction) {
        size_t t = res.cx;
        res.cx = res.cy;
        res.cy = t;
    }
    res.start = cfg->len * part;
    res.stop = res.start + part;
    return res;
}

 void sq_init(SmallQuery* sq) {
    sq->size = sq->start = 0;
}


 int sq_push(SmallQuery* sq, size_t x, size_t y) {
    sq->x[(sq->start + sq->size) % 10] = x;
    sq->y[(sq->start + sq->size) % 10] = y;
    sq->size++;
    if (sq->size > 10) return 1;
    else return 0;
}


 int sq_pop(SmallQuery* sq, size_t* x, size_t* y) {
    if (sq->size == 0) return 1;
    *x = sq->x[sq->start];
    *y = sq->y[sq->start];
    sq->start = (sq->start + 1) % 10;
    sq->size--;
    return 0;
}

 int sq_contains(SmallQuery* sq, size_t x, size_t y) {
    for (int i = 0; i < sq->size; i++) {
        int j = (i + sq->start) / 10;
        if (sq->x[j] == x) {
            if (sq->y[j] == y) return 1;
        }
    }
    return 0;
}

#define insert_correct(x, y) if (x >= 0 && x < plane->actualWidth && y >= 0 && y < plane->height && !sq_contains(sq, x, y) && pixel(plane, x, y) == color) sq_push(sq, x, y);

 int traverse(PlanePart* plane, SmallQuery* sq, size_t len, size_t* cross_x, size_t* cross_y, plane_t color) {
    size_t x, y;
    while(!sq_pop(sq, &x, &y)) {
        insert_correct(x - 1, y - 1);
        insert_correct(x - 1, y);
        insert_correct(x - 1, y + 1);
        insert_correct(x, y - 1);
        insert_correct(x, y + 1);
        insert_correct(x + 1, y - 1);
        insert_correct(x + 1, y);
        insert_correct(x + 1, y + 1);
        if (x % len == 0 || y % len == 0) {
            *cross_x = x;
            *cross_y = y;
            return 1;
        }
    }
    return 0;
}

 int should_count(size_t x, size_t y, LineConfig* cfg, size_t len) {
    size_t line_x, line_y;
    if (cfg->direction == 0) {
        line_y = cfg->cy;
        line_x = cfg->stop;
    } else {
        line_x = cfg->cx;
        line_y = cfg->stop;
    }
    int dx = x - line_x;
    int dy = y - line_y;
    
    if (dx % (len << cfg->rang) == 0 || dy % (len << cfg->rang)) {
        return 0;
    }
    if (dy > 0) return 0; 
    if (dy < 0) return 1; 
    if (dx > 0) return 0; 
    return 1;
}



