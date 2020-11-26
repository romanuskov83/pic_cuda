//
// Created by romanu on 17.11.20.
//
#include <assert.h>
#include "grid.h"

int findCell(Coordinate r, Coordinate *grid, int start, int end);


int findCell(Coordinate r, Coordinate *grid, int start, int end) {
    assert(grid[start] <= r);
    assert(grid[end+1] >= r);
    if(start == end)
        return start;

    int middle = (start+end)/2;
    if(grid[middle+1] >= r)
        return findCell(r,grid,start,middle);
    else
        return findCell(r,grid,middle+1,end);
}

int findCell(Coordinate r, Coordinate* gridData, int gridDim) {
    int start = 0;
    int end = gridDim-1;
    return findCell(r,gridData,start,end);
}

