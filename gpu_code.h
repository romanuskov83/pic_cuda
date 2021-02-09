#ifndef __GPU_CODE__H__PIC__
#define __GPU_CODE__H__PIC__

#include "common.h"
#define EXCHANGE_PARTICLES_COUNT (1024*1024)



struct CUstream_st;

typedef struct {
    int deviceId;
    int warpSize;

    //grid
    int gridNx;
    int gridNy;
    int gridNz;
    int gridDim;
    Coordinate* gridData;
    Coordinate* gridX;
    Coordinate* gridY;
    Coordinate* gridZ;
    FieldComponent* electricData;
    FieldComponent* magneticData;
    FieldComponent* currentData;
    FieldComponent* currentDataHost;

    //Particles
    ParticlesBlock* particlesBlocks;
    int particlesBlocksCount;

    ParticlesBlock* exchangeParticlesBlocks;
    ParticlesBlock* exchangeParticlesBlocksHost;
    int* exchangeCounters;
    float *position;

#ifdef CELL_BOUND_BLOCKS
    unsigned int* startBlock;
    unsigned int* endBlock;
    unsigned int* particlesCounter;
#else
    int workingBlocksCount;
#endif

    CUstream_st* modellingStream;
    CUstream_st* exchangesStream;

} GpuState;


GpuState* gpuInit(int deviceId, int gridNx, int gridNy, int gridNz, Coordinate* gridX, Coordinate* gridY, Coordinate* gridZ);
bool gpuUpdateFieldsData(GpuState* gpuState,FieldComponent* electricData, FieldComponent* magneticData);
bool gpuExchangeParticles(GpuState *state, ParticleInfo *particles, int& countIn, int& countOut);
bool gpuDumpParticles(GpuState *state, ParticleInfo *particles, int& countOut);
bool gpuMakeStep(GpuState* state,float startTime, float endTime);
#endif