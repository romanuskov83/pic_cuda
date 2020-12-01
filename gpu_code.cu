#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include <driver_types.h>
#include "gpu_code.h"
#include "grid.h"
#define tx threadIdx.x
#define bx blockIdx.x

#define GPU_RESOURCE_UTILIZATION_FACTOR 0.9f

#define EXCHANGE_PARTICLES_BLOCKS (EXCHANGE_PARTICLES_COUNT/PARTICLE_BLOCK_SIZE)


GpuState* gpuInit(int deviceId, int gridNx, int gridNy, int gridNz, Coordinate *gridX, Coordinate *gridY, Coordinate *gridZ) {
    GpuState* result = new GpuState;

    result->deviceId = deviceId;
    result->gridNx = gridNx;
    result->gridNy = gridNy;
    result->gridNz = gridNz;
    result->gridX = gridX;
    result->gridY = gridY;
    result->gridZ = gridZ;

    int gridDim = gridNx*gridNy*gridNz;
    result->gridDim = gridDim;
    int gridPlusOneDim = (gridNx+1)*(gridNy+1)*(gridNz+1);

    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(deviceId);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?";
        return NULL;
    }

    struct cudaDeviceProp props;
    cudaStatus = cudaGetDeviceProperties(&props, deviceId);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed! ";
        return NULL;
    }
    std::cout << "Using " << props.name << " with " << props.multiProcessorCount << " MP\n";

    cudaStatus = cudaStreamCreate(&result->modellingStream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaCreateStream failed! modellingStream";
        return NULL;
    }

    cudaStatus = cudaStreamCreate(&result->exchangesStream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaCreateStream failed! exchangesStream";
        return NULL;
    }

    result->workingBlocksCount = props.multiProcessorCount*2;
    result->particlesBlocksCount = (int)((props.totalGlobalMem*GPU_RESOURCE_UTILIZATION_FACTOR
                                          -gridDim*3*sizeof(FieldComponent)-gridPlusOneDim*3*sizeof(FieldComponent)
                                          -(gridNx+1+gridNy+1+gridNz+1)*sizeof(Coordinate)
                                          -sizeof(ParticlesBlock)*EXCHANGE_PARTICLES_BLOCKS-gridDim*12*sizeof(FieldComponent)*result->workingBlocksCount)/sizeof(ParticlesBlock));
    std::cout << "Particles blocks count: " << result->particlesBlocksCount << "\n";



    cudaStatus = cudaMallocManaged(&result->exchangeCounters, 3*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed! exchangeCounters");
        return NULL;
    }


    cudaStatus = cudaMalloc(&result->gridData, (gridNx+gridNy+gridNz+3)*sizeof(Coordinate));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! gridData");
        return NULL;
    }

    cudaStatus = cudaMemcpy(result->gridData, gridX, (gridNx+1)*sizeof(Coordinate),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! gridX");
        return NULL;
    }

    cudaStatus = cudaMemcpy(&result->gridData[gridNx+1], gridY, (gridNy+1)*sizeof(Coordinate),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! gridY");
        return NULL;
    }

    cudaStatus = cudaMemcpy(&result->gridData[gridNx+1+gridNy+1], gridZ, (gridNz+1)*sizeof(Coordinate),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! gridZ");
        return NULL;
    }


    cudaStatus = cudaMalloc(&result->particlesBlocks, sizeof(ParticlesBlock)*result->particlesBlocksCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! particlesBlocks");
        return NULL;
    }

    cudaStatus = cudaMemset(result->particlesBlocks, 0x0, sizeof(ParticlesBlock)*result->particlesBlocksCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemSet failed! particlesBlocks");
        return NULL;
    }

    cudaStatus = cudaMalloc(&result->exchangeParticlesBlocks, sizeof(ParticlesBlock)*EXCHANGE_PARTICLES_BLOCKS);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! exchangeParticlesBlocks");
        return NULL;
    }


    result->exchangeParticlesBlocksHost = (ParticlesBlock*)malloc(sizeof(ParticlesBlock)*EXCHANGE_PARTICLES_BLOCKS);
    if (result->exchangeParticlesBlocksHost == NULL) {
        fprintf(stderr, "malloc failed! exchangeParticlesBlocksHost");
        return NULL;

    }


    cudaStatus = cudaMalloc(&result->electricData, sizeof(FieldComponent)*((gridNx)*(gridNy+1)*(gridNz+1) + (gridNx+1)*(gridNy)*(gridNz+1) + (gridNx+1)*(gridNy+1)*(gridNz)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! electricData");
        return NULL;
    }

    cudaStatus = cudaMalloc(&result->magneticData, sizeof(FieldComponent)*((gridNx+1)*(gridNy)*(gridNz) + (gridNx)*(gridNy+1)*(gridNz) + (gridNx)*(gridNy)*(gridNz+1)));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! magneticData");
        return NULL;
    }


    result->currentDataHost = (FieldComponent *)malloc(sizeof(FieldComponent)*gridDim*12);

    cudaStatus = cudaMalloc(&result->currentData, sizeof(FieldComponent)*gridDim*12*result->workingBlocksCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! currentData");
        return NULL;
    }



    return result;
}

bool gpuUpdateFieldsData(GpuState* gpuState,FieldComponent* electricData, FieldComponent* magneticData) {

    cudaError_t cudaStatus;

    cudaStatus = cudaMemcpy(gpuState->electricData, electricData, sizeof(FieldComponent)*((gpuState->gridNx)*(gpuState->gridNy+1)*(gpuState->gridNz+1) + (gpuState->gridNx+1)*(gpuState->gridNy)*(gpuState->gridNz+1) + (gpuState->gridNx+1)*(gpuState->gridNy+1)*(gpuState->gridNz)),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! electricData");
        return false;
    }

    cudaStatus = cudaMemcpy(gpuState->magneticData, magneticData, sizeof(FieldComponent)*((gpuState->gridNx+1)*(gpuState->gridNy)*(gpuState->gridNz) + (gpuState->gridNx)*(gpuState->gridNy+1)*(gpuState->gridNz) + (gpuState->gridNx)*(gpuState->gridNy)*(gpuState->gridNz+1)),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! magneticData");
        return false;
    }

    cudaStatus = cudaMemset(gpuState->currentData, 0x0, sizeof(FieldComponent)*gpuState->gridNx*gpuState->gridNy*gpuState->gridNz*12);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed! currentData");
        return false;
    }


    return true;
}



/*Exchanges stream*/

__global__ __launch_bounds__(1024, 1) void exchangeParticles(ParticlesBlock* modellingBlocks, int modellingBlocksCount, ParticlesBlock* exchangeBlocks, int* countersGlobal) {
    __shared__ int counters[2];
    if(tx == 0) {
        counters[0] = *countersGlobal;
        counters[1] = 0;
    }

    int totalModelling = modellingBlocksCount*PARTICLE_BLOCK_SIZE;
    int iterations = totalModelling/blockDim.x;

    /* Put new particles in */
    for(int i = 0; i < iterations; i++) {
        int pIdx = i*blockDim.x + tx;
        int bi = pIdx/PARTICLE_BLOCK_SIZE;
        int pi = pIdx%PARTICLE_BLOCK_SIZE;
        int cellIdFlag = modellingBlocks[bi].cellIdFlag[pi];
        if((cellIdFlag & FLAG_MASK) == FLAG_DIED) {
            int old = atomicAdd(counters,-1);
            if(old > 0) {
                int bi2 = (old-1)/PARTICLE_BLOCK_SIZE;
                int pi2 = (old-1)%PARTICLE_BLOCK_SIZE;
                modellingBlocks[bi].rx[pi] = exchangeBlocks[bi2].rx[pi2];
                modellingBlocks[bi].ry[pi] = exchangeBlocks[bi2].ry[pi2];
                modellingBlocks[bi].rz[pi] = exchangeBlocks[bi2].rz[pi2];

                modellingBlocks[bi].px[pi] = exchangeBlocks[bi2].px[pi2];
                modellingBlocks[bi].py[pi] = exchangeBlocks[bi2].py[pi2];
                modellingBlocks[bi].pz[pi] = exchangeBlocks[bi2].pz[pi2];

                modellingBlocks[bi].weight[pi] = exchangeBlocks[bi2].weight[pi2];
                modellingBlocks[bi].currentTime[pi] = exchangeBlocks[bi2].currentTime[pi2];

                modellingBlocks[bi].cellIdFlag[pi] = exchangeBlocks[bi2].cellIdFlag[pi2];
                modellingBlocks[bi].id[pi] = exchangeBlocks[bi2].id[pi2];
            }
        }

    }
    __syncthreads();

    /* not all IN particles were processed - no space - so we take OUT particles starting from Nth position keeping N IN particles for later*/
    if(counters[0] > 0) {
        counters[1] = counters[0];
    }

    __syncthreads();

    /* Put new particles in */
    for(int i = 0; i < iterations; i++) {
        int pIdx = i*blockDim.x + tx;
        int bi = pIdx/PARTICLE_BLOCK_SIZE;
        int pi = pIdx%PARTICLE_BLOCK_SIZE;
        if((modellingBlocks[bi].cellIdFlag[pi] & FLAG_MASK) > FLAG_OK) {
            int old = atomicAdd(counters+1,1);
            if(old < EXCHANGE_PARTICLES_BLOCKS*PARTICLE_BLOCK_SIZE) {
                int bi2 = old/PARTICLE_BLOCK_SIZE;
                int pi2 = old%PARTICLE_BLOCK_SIZE;
                exchangeBlocks[bi2].rx[pi2] = modellingBlocks[bi].rx[pi];
                exchangeBlocks[bi2].ry[pi2] = modellingBlocks[bi].ry[pi];
                exchangeBlocks[bi2].rz[pi2] = modellingBlocks[bi].rz[pi];

                exchangeBlocks[bi2].px[pi2] = modellingBlocks[bi].px[pi];
                exchangeBlocks[bi2].py[pi2] = modellingBlocks[bi].py[pi];
                exchangeBlocks[bi2].pz[pi2] = modellingBlocks[bi].pz[pi];

                exchangeBlocks[bi2].currentTime[pi2] = modellingBlocks[bi].currentTime[pi];
                exchangeBlocks[bi2].weight[pi2] = modellingBlocks[bi].weight[pi];
                exchangeBlocks[bi2].cellIdFlag[pi2] = modellingBlocks[bi].cellIdFlag[pi];
                exchangeBlocks[bi2].id[pi2] = modellingBlocks[bi].id[pi];

                modellingBlocks[bi].cellIdFlag[pi] = FLAG_DIED;
            }
        }
    }
    __syncthreads();
    if(tx == 0) {
        countersGlobal[0] = counters[0];
        countersGlobal[1] = counters[1];
    }
}

bool gpuExchangeParticles(GpuState *state, ParticleInfo *particles, int& countIn, int& countOut) {


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaError_t cudaStatus;

    for(int i = 0; i < countIn; i++) {
        ParticleInfo& info = particles[i];
        int ix = findCell(info.rx,state->gridX, state->gridNx);
        int iy = findCell(info.ry,state->gridY, state->gridNy);
        int iz = findCell(info.rz,state->gridZ, state->gridNz);
        int cellIdx = cel_idx(ix,iy,iz,state->gridNx,state->gridNy,state->gridNz);
        int pi = i%PARTICLE_BLOCK_SIZE;

        ParticlesBlock* pb = &state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE];

        pb->rx[pi] = particles[i].rx;
        pb->ry[pi] = particles[i].ry;
        pb->rz[pi] = particles[i].rz;

        pb->px[pi] = particles[i].px;
        pb->py[pi] = particles[i].py;
        pb->pz[pi] = particles[i].pz;
        pb->id[pi] = particles[i].id;
        pb->currentTime[pi] = particles[i].currentTime;

        pb->weight[pi] = particles[i].weight;

        pb->cellIdFlag[pi] = FLAG_OK | cellIdx;
    }

    state->exchangeCounters[0] = countIn;
    state->exchangeCounters[1] = 0;


    cudaEventRecord(start);

    cudaStatus = cudaMemcpyAsync(state->exchangeParticlesBlocks,state->exchangeParticlesBlocksHost, sizeof(ParticlesBlock)*EXCHANGE_PARTICLES_BLOCKS,cudaMemcpyHostToDevice,state->exchangesStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! exchangeParticlesBlocks");
        return false;
    }

    cudaStatus = cudaStreamSynchronize(state->exchangesStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! exchangesStream");
        return false;
    }

    dim3 blocks;
    blocks.x = 1;
    blocks.y = 1;
    blocks.z = 1;
    dim3 threads;
    threads.x = 1024;
    threads.y = 1;
    threads.z = 1;
    //std::cout << "exchange particles kernel\n";
    exchangeParticles<<<blocks,threads,0,state->exchangesStream>>>(state->particlesBlocks, state->particlesBlocksCount,state->exchangeParticlesBlocks, state->exchangeCounters);
    cudaStatus = cudaStreamSynchronize(state->exchangesStream);
    if(cudaStatus != cudaSuccess) {
        std::cerr << "Kernel failed: exchangeParticles " << cudaStatus <<"\n";
        return false;
    }



    printf("AVAILABLE PARTICLES %d PARTICLES TO SEND %d\n",-state->exchangeCounters[0], state->exchangeCounters[1] - (state->exchangeCounters[0] > 0 ? PARTICLE_BLOCK_SIZE*EXCHANGE_PARTICLES_BLOCKS-state->exchangeCounters[0]: 0));


    cudaStatus = cudaMemcpyAsync(state->exchangeParticlesBlocksHost,state->exchangeParticlesBlocks, sizeof(ParticlesBlock)*EXCHANGE_PARTICLES_BLOCKS,cudaMemcpyDeviceToHost,state->exchangesStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! exchangeParticlesBlocks");
        return false;
    }

    cudaStatus = cudaStreamSynchronize(state->exchangesStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! exchangesStream");
        return false;
    }


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "EXCHANGEs DONE IN " << milliseconds << "ms\n";


    for(int i = 0; i < state->exchangeCounters[1]; i++) {
        particles[i].rx = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].rx[i%PARTICLE_BLOCK_SIZE];
        particles[i].ry = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].ry[i%PARTICLE_BLOCK_SIZE];
        particles[i].rz = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].rz[i%PARTICLE_BLOCK_SIZE];

        particles[i].px = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].px[i%PARTICLE_BLOCK_SIZE];
        particles[i].py = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].py[i%PARTICLE_BLOCK_SIZE];
        particles[i].pz = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].pz[i%PARTICLE_BLOCK_SIZE];

        particles[i].weight = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].weight[i%PARTICLE_BLOCK_SIZE];
        particles[i].currentTime = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].currentTime[i%PARTICLE_BLOCK_SIZE];
        particles[i].cellIdFlag = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].cellIdFlag[i%PARTICLE_BLOCK_SIZE];
        particles[i].id = state->exchangeParticlesBlocksHost[i/PARTICLE_BLOCK_SIZE].id[i%PARTICLE_BLOCK_SIZE];

    }

    countIn = state->exchangeCounters[0] > 0 ? state->exchangeCounters[0] : 0;
    countOut = state->exchangeCounters[1];
    return true;
}


/*Modelling stream*/




//__launch_bounds__(PARTICLE_BLOCK_SIZE, 16)

__global__  void makeStep(ParticlesBlock* particlesBlocks, int particlesBlocksCount,const FieldComponent* eX, const FieldComponent* hX, FieldComponent* currentData,const float startTime, const float endTime,const int gridNx, const int gridNy, const int gridNz, const Coordinate* gridDataX, int* exchangeCounters
)
{

    int iterations = (PARTICLE_BLOCK_SIZE*particlesBlocksCount)/(blockDim.x*gridDim.x);
    int i;
    int pIdx;
    int bi=0;
    int pi=0;

    Coordinate rx;
    Coordinate ry;
    Coordinate rz;
    Coordinate px;
    Coordinate py;
    Coordinate pz;
    float currentTime=0;

    for(i = 0; i < iterations; ++i) {
        pIdx = blockDim.x*gridDim.x*i + blockDim.x*bx+tx;
        bi = pIdx/PARTICLE_BLOCK_SIZE;
        pi = pIdx%PARTICLE_BLOCK_SIZE;
        int flag = particlesBlocks[bi].cellIdFlag[pi];
        if((flag & FLAG_MASK) == FLAG_OK) {
            currentTime = particlesBlocks[bi].currentTime[pi];
            if(currentTime < endTime && currentTime >= startTime)
                break;
        }
    }


    __syncthreads();

    if(i == iterations) {
        return;
    }


    exchangeCounters[2] = 1;

    const int cellId = particlesBlocks[bi].cellIdFlag[pi] & CELL_ID_MASK;
    const int ix = x_idx(cellId,gridNx,gridNy,gridNz);
    const int iy = y_idx(cellId,gridNx,gridNy,gridNz);
    const int iz = z_idx(cellId,gridNx,gridNy,gridNz);

    const Coordinate* gridDataY = gridDataX+gridNx+1;
    const Coordinate* gridDataZ = gridDataX+gridNx+gridNy+2;

    const FieldComponent* eY = eX+gridNx*(gridNy+1)*(gridNz+1);
    const FieldComponent* eZ = eX+gridNx*(gridNy+1)*(gridNz+1)+(gridNx+1)*gridNy*(gridNz+1);

    const FieldComponent* hY = hX+(gridNx+1)*gridNy*gridNz;
    const FieldComponent* hZ = hX+(gridNx+1)*gridNy*gridNz+gridNx*(gridNy+1)*gridNz;


    const Coordinate fromX = gridDataX[ix];
    const Coordinate toX = gridDataX[ix+1];
    const Coordinate fromY = gridDataY[iy];
    const Coordinate toY = gridDataY[iy+1];
    const Coordinate fromZ = gridDataZ[iz];
    const Coordinate toZ = gridDataZ[iz+1];

    rx = particlesBlocks[bi].rx[pi];
    ry = particlesBlocks[bi].ry[pi];
    rz = particlesBlocks[bi].rz[pi];

    px = particlesBlocks[bi].px[pi];
    py = particlesBlocks[bi].py[pi];
    pz = particlesBlocks[bi].pz[pi];

    Coordinate  u02= px*px + py*py + pz*pz;
    Coordinate  q_v = C/sqrtf(1.0f+u02);
    Coordinate vx = px*q_v;
    Coordinate vy = py*q_v;
    Coordinate vz = pz*q_v;
//    Coordinate v = norm3df(vx,vy,vz);
    


    Coordinate toBorderTimeX;
    if(vx > 0) {
        toBorderTimeX = (toX-rx)/vx;
    } else if(vx < 0) {
        toBorderTimeX = (fromX-rx)/vx;
    } else {
        toBorderTimeX = 1e9;
    }


    Coordinate toBorderTimeY;
    if(vy > 0) {
        toBorderTimeY = (toY-ry)/vy;
    } else if(vy < 0) {
        toBorderTimeY = (fromY-ry)/vy;
    } else {
        toBorderTimeY = 1e9;
    }

    Coordinate toBorderTimeZ;
    if(vz > 0) {
        toBorderTimeZ = (toZ-rz)/vz;
    } else if(vz < 0) {
        toBorderTimeZ = (fromZ-rz)/vz;
    } else {
        toBorderTimeZ = 1e9;
    }

    Coordinate toBorderTime = min(toBorderTimeX,min(toBorderTimeY,toBorderTimeZ));
    const Coordinate time = min(toBorderTime,endTime-currentTime);

    //currents_1(Me*Qe*particlesBlocks[bi].weight[pi],ix,iy,iz,rx,ry,rz,vx*time,vy*time,vz*time);

    //!коэффициенты для целых точек
    float cx =(rx-fromX)/(toX-fromX);
    float cy =(ry-fromY)/(toY-fromY);
    float cz =(rz-fromZ)/(toZ-fromZ);

    int ixc;
    int iyc=0;
    int izc=0;
    float cxc;
    float cyc=0;
    float czc=0;

    if(rx < (fromX+toX)/2) {
        if(ix == 0) {
            ixc = ix;
            cxc = 0.0f;
        } else {
            ixc = ix-1;
            cxc = (rx - (gridDataX[ixc]+fromX)/2)/((fromX+toX)/2-(gridDataX[ixc]+fromX)/2);
        }
    } else {
        if(ix == gridNx-1) {
            ixc = ix-1;
            cxc = 1.0f;
        } else {
            ixc = ix;
            cxc = (rx - (fromX+toX)/2)/((gridDataX[ixc+2]+toX)/2-(fromX+toX)/2);
        }
    };


    if(ry < (fromY+toY)/2) {
        if(iy == 0) {
            iyc = iy;
            cyc = 0.0f;
        } else {
            iyc = iy-1;
            cyc = (ry - (gridDataY[iyc]+fromY)/2)/((fromY+toY)/2-(gridDataY[iyc]+fromY)/2);
        }
    } else {
        if(iy == gridNy-1) {
            iyc = iy-1;
            cyc = 1.0f;
        } else {
            iyc = iy;
            cyc = (ry - (fromY+toY)/2)/((gridDataY[iyc+2]+toY)/2-(fromY+toY)/2);
        }
    };

    if(rz < (fromZ+toZ)/2) {
        if(iz == 0) {
            izc = iz;
            czc = 0.0f;
        } else {
            izc = iz-1;
            czc = (rz - (gridDataZ[izc]+fromZ)/2)/((fromZ+toZ)/2-(gridDataZ[izc]+fromZ)/2);
        }
    } else {
        if(iz == gridNz-1) {
            izc = iz-1;
            czc = 1.0f;
        } else {
            izc = iz;
            czc = (rz - (fromZ+toZ)/2)/((gridDataZ[izc+2]+toZ)/2-(fromZ+toZ)/2);
        }
    };


//    printf("ZZZZ %f\n",(fromX+toX)/2);
//    printf("ZZZZ %f\n",(gridDataX[ix+2]+toX)/2);
//    printf("\t\t%f %f %f - %d %d %d / %f %f %f - %d %d %d / %f %f %f\n", rx,ry,rz,ix,iy,iz,cx,cy,cz,ixc,iyc,izc,cxc,cyc,czc);

    if(toBorderTime > endTime-currentTime) {
        rx += vx*time;
        ry += vy*time;
        rz += vz*time;
        currentTime = endTime;

    } else {
        //printf("REACHED BORDER\n");

        rx += vx*toBorderTime;
        ry += vy*toBorderTime;
        rz += vz*toBorderTime;
        currentTime += toBorderTime;

        if(toBorderTime == toBorderTimeX) {
            if(px > 0) {
                rx = toX;
                if(ix == gridNx-1) {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & CELL_ID_MASK) | FLAG_FLEW_PLUS_X;
                } else {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & FLAG_MASK) | (cellId+gridNz*gridNy);
                }
            } else {
                rx = fromX;
                if(ix == 0) {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & CELL_ID_MASK) | FLAG_FLEW_MINUS_X;
                } else {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & FLAG_MASK) | (cellId-gridNz*gridNy);
                }
            }

        } else if(toBorderTime == toBorderTimeY) {
            if(py > 0) {
                ry = toY;
                if(iy == gridNy-1) {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & CELL_ID_MASK) | FLAG_FLEW_PLUS_Y;
                } else {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & FLAG_MASK) | (cellId+gridNz);
                }
            } else {
                ry = fromY;
                if(iy == 0) {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & CELL_ID_MASK) | FLAG_FLEW_MINUS_Y;
                } else {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & FLAG_MASK) | (cellId-gridNz);
                }
            }
        } else {
            if(pz > 0) {
                rz = toZ;
                if(iz == gridNz-1) {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & CELL_ID_MASK) | FLAG_FLEW_PLUS_Z;
                } else {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & FLAG_MASK) | (cellId+1);
                }
            } else {
                rz = fromZ;
                if(iz == 0) {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & CELL_ID_MASK) | FLAG_FLEW_MINUS_Z;
                } else {
                    particlesBlocks[bi].cellIdFlag[pi] = (particlesBlocks[bi].cellIdFlag[pi] & FLAG_MASK) | (cellId-1);
                }
            }
        }

    }


    const float cfield = -Qe*time*Ee/(C*Me);


    const float yex=cfield*(eX[cel_idx(ixc,iy,iz,gridNx,gridNy+1,gridNz+1)]*(1.0f-cxc)*(1.0f-cy)*(1.0f-cz) +
                      eX[cel_idx(ixc+1,iy,iz,gridNx,gridNy+1,gridNz+1)]*(cxc)*(1.0f-cy)*(1.0f-cz) +
                      eX[cel_idx(ixc,iy+1,iz,gridNx,gridNy+1,gridNz+1)]*(1.0f-cxc)*(cy)*(1.0f-cz) +
                      eX[cel_idx(ixc+1,iy+1,iz,gridNx,gridNy+1,gridNz+1)]*(cxc)*(cy)*(1.0f-cz) +
                      eX[cel_idx(ixc,iy,iz+1,gridNx,gridNy+1,gridNz+1)]*(1.0f-cxc)*(1.0f-cy)*(cz) +
                      eX[cel_idx(ixc+1,iy,iz+1,gridNx,gridNy+1,gridNz+1)]*(cxc)*(1.0f-cy)*(cz) +
                      eX[cel_idx(ixc,iy+1,iz+1,gridNx,gridNy+1,gridNz+1)]*(1.0f-cxc)*(cy)*(cz) +
                      eX[cel_idx(ixc+1,iy+1,iz+1,gridNx,gridNy+1,gridNz+1)]*(cxc)*(cy)*(cz));


    const float yey=cfield*(eY[cel_idx(ix,iyc,iz,gridNx+1,gridNy,gridNz+1)]*(1.0f-cx)*(1.0f-cyc)*(1.0f-cz) +
                      eY[cel_idx(ix+1,iyc,iz,gridNx+1,gridNy,gridNz+1)]*(cx)*(1.0f-cyc)*(1.0f-cz) +
                      eY[cel_idx(ix,iyc+1,iz,gridNx+1,gridNy,gridNz+1)]*(1.0f-cx)*(cyc)*(1.0f-cz) +
                      eY[cel_idx(ix+1,iyc+1,iz,gridNx+1,gridNy,gridNz+1)]*(cx)*(cyc)*(1.0f-cz) +
                      eY[cel_idx(ix,iyc,iz+1,gridNx+1,gridNy,gridNz+1)]*(1.0f-cx)*(1.0f-cyc)*(cz) +
                      eY[cel_idx(ix+1,iyc,iz+1,gridNx+1,gridNy,gridNz+1)]*(cx)*(1.0f-cyc)*(cz) +
                      eY[cel_idx(ix,iyc+1,iz+1,gridNx+1,gridNy,gridNz+1)]*(1.0f-cx)*(cyc)*(cz) +
                      eY[cel_idx(ix+1,iyc+1,iz+1,gridNx+1,gridNy,gridNz+1)]*(cx)*(cyc)*(cz));

    const float yez=cfield*(eZ[cel_idx(ix,iy,izc,gridNx+1,gridNy+1,gridNz)]*(1.0f-cx)*(1.0f-cy)*(1.0f-czc) +
                      eZ[cel_idx(ix+1,iy,izc,gridNx+1,gridNy+1,gridNz)]*(cx)*(1.0f-cy)*(1.0f-czc) +
                      eZ[cel_idx(ix,iy+1,izc,gridNx+1,gridNy+1,gridNz)]*(1.0f-cx)*(cy)*(1.0f-czc) +
                      eZ[cel_idx(ix+1,iy+1,izc,gridNx+1,gridNy+1,gridNz)]*(cx)*(cy)*(1.0f-czc) +
                      eZ[cel_idx(ix,iy,izc+1,gridNx+1,gridNy+1,gridNz)]*(1.0f-cx)*(1.0f-cy)*(czc) +
                      eZ[cel_idx(ix+1,iy,izc+1,gridNx+1,gridNy+1,gridNz)]*(cx)*(1.0f-cy)*(czc) +
                      eZ[cel_idx(ix,iy+1,izc+1,gridNx+1,gridNy+1,gridNz)]*(1.0f-cx)*(cy)*(czc) +
                      eZ[cel_idx(ix+1,iy+1,izc+1,gridNx+1,gridNy+1,gridNz)]*(cx)*(cy)*(czc));

    float yhx=cfield*(hX[cel_idx(ix,iyc,izc,gridNx+1,gridNy,gridNz)]*(1.0f-cx)*(1.0f-cyc)*(1.0f-czc)+
                      hX[cel_idx(ix+1,iyc,izc,gridNx+1,gridNy,gridNz)]*(cx)*(1.0f-cyc)*(1.0f-czc)+
                      hX[cel_idx(ix,iyc+1,izc,gridNx+1,gridNy,gridNz)]*(1.0f-cx)*(cyc)*(1.0f-czc)+
                      hX[cel_idx(ix+1,iyc+1,izc,gridNx+1,gridNy,gridNz)]*(cx)*(cyc)*(1.0f-czc)+
                      hX[cel_idx(ix,iyc,izc+1,gridNx+1,gridNy,gridNz)]*(1.0f-cx)*(1.0f-cyc)*(czc)+
                      hX[cel_idx(ix+1,iyc,izc+1,gridNx+1,gridNy,gridNz)]*(cx)*(1.0f-cyc)*(czc)+
                      hX[cel_idx(ix,iyc+1,izc+1,gridNx+1,gridNy,gridNz)]*(1.0f-cx)*(cyc)*(czc)+
                      hX[cel_idx(ix+1,iyc+1,izc+1,gridNx+1,gridNy,gridNz)]*(cx)*(cyc)*(czc));

    float yhy=cfield*(hY[cel_idx(ixc,iy,izc,gridNx,gridNy+1,gridNz)]*(1.0f-cxc)*(1.0f-cy)*(1.0f-czc)+
                      hY[cel_idx(ixc+1,iy,izc,gridNx,gridNy+1,gridNz)]*(cxc)*(1.0f-cy)*(1.0f-czc)+
                      hY[cel_idx(ixc,iy+1,izc,gridNx,gridNy+1,gridNz)]*(1.0f-cxc)*(cy)*(1.0f-czc)+
                      hY[cel_idx(ixc+1,iy+1,izc,gridNx,gridNy+1,gridNz)]*(cxc)*(cy)*(1.0f-czc)+
                      hY[cel_idx(ixc,iy,izc+1,gridNx,gridNy+1,gridNz)]*(1.0f-cxc)*(1.0f-cy)*(czc)+
                      hY[cel_idx(ixc+1,iy,izc+1,gridNx,gridNy+1,gridNz)]*(cxc)*(1.0f-cy)*(czc)+
                      hY[cel_idx(ixc,iy+1,izc+1,gridNx,gridNy+1,gridNz)]*(1.0f-cxc)*(cy)*(czc)+
                      hY[cel_idx(ixc+1,iy+1,izc+1,gridNx,gridNy+1,gridNz)]*(cxc)*(cy)*(czc));
    
    float yhz=cfield*(hZ[cel_idx(ixc,iyc,iz,gridNx,gridNy,gridNz+1)]*(1.0f-cxc)*(1.0f-cyc)*(1.0f-cz)+
                      hZ[cel_idx(ixc+1,iyc,iz,gridNx,gridNy,gridNz+1)]*(cxc)*(1.0f-cyc)*(1.0f-cz)+
                      hZ[cel_idx(ixc,iyc+1,iz,gridNx,gridNy,gridNz+1)]*(1.0f-cxc)*(cyc)*(1.0f-cz)+
                      hZ[cel_idx(ixc+1,iyc+1,iz,gridNx,gridNy,gridNz+1)]*(cxc)*(cyc)*(1.0f-cz)+
                      hZ[cel_idx(ixc,iyc,iz+1,gridNx,gridNy,gridNz+1)]*(1.0f-cxc)*(1.0f-cyc)*(cz)+
                      hZ[cel_idx(ixc+1,iyc,iz+1,gridNx,gridNy,gridNz+1)]*(cxc)*(1.0f-cyc)*(cz)+
                      hZ[cel_idx(ixc,iyc+1,iz+1,gridNx,gridNy,gridNz+1)]*(1.0f-cxc)*(cyc)*(cz)+
                      hZ[cel_idx(ixc+1,iyc+1,iz+1,gridNx,gridNy,gridNz+1)]*(cxc)*(cyc)*(cz));

    printf("%f %f %f\n",rx,ry,rz);

    //!ux,uy,uz calculation
    px-=yex;
    py-=yey;
    pz-=yez;

    const double d2gm=0.5/sqrtf(1.0f + 0.25f*sqr(2*px+yex) + 0.25f*sqr(2*py+yey) + 0.25f*sqr(2*pz+yez));


    const double _bx=yhx*d2gm;
    const double _by=yhy*d2gm;
    const double _bz=yhz*d2gm;

    const double fx=px-(yex+(yhz*py-yhy*pz)*d2gm);
    const double fy=py-(yey+(yhx*pz-yhz*px)*d2gm);
    const double fz=pz-(yez+(yhy*px-yhx*py)*d2gm);
    const double bx2=_bx*_bx;
    const double by2=_by*_by;
    const double bz2=_bz*_bz;
    const double bxy=_bx*_by;
    const double bxz=_bx*_bz;
    const double byz=_by*_bz;

    const double ddt=1.0f/(1.0f+bx2+by2+bz2);
    //printf("AAA %le\n",ddt);

    px=ddt*(fx*(1.0+bx2)+fy*(bxy-_bz)  +fz*(bxz+_by));
    py=ddt*(fx*(bxy+_bz)  +fy*(1.0+by2)+fz*(byz-_bx));
    pz=ddt*(fx*(bxz-_by)  +fy*(byz+_bx)  +fz*(1.0+bz2));


    particlesBlocks[bi].rx[pi] = rx;
    particlesBlocks[bi].ry[pi] = ry;
    particlesBlocks[bi].rz[pi] = rz;

    particlesBlocks[bi].px[pi] = px;
    particlesBlocks[bi].py[pi] = py;
    particlesBlocks[bi].pz[pi] = pz;

    particlesBlocks[bi].currentTime[pi] = currentTime;


    const int offset = 12*(bx*gridNx*gridNy*gridNz+cellId);

    atomicAdd(currentData+offset,px);
    atomicAdd(currentData+offset+1,py);
    atomicAdd(currentData+offset+2,pz);

    atomicAdd(currentData+offset+3,px);
    atomicAdd(currentData+offset+4,py);
    atomicAdd(currentData+offset+5,pz);

    atomicAdd(currentData+offset+6,px);
    atomicAdd(currentData+offset+7,py);
    atomicAdd(currentData+offset+8,pz);

    atomicAdd(currentData+offset+9,px);
    atomicAdd(currentData+offset+10,py);
    atomicAdd(currentData+offset+11,pz);

}

__global__  void collectCurrentData(FieldComponent* currentData, int dataRes, int copiesCount)
{
    int i = bx*blockDim.x+tx;
    if(i >= dataRes)
        return;
    for(int j = 1; j < copiesCount;j++)
    {
        currentData[i] += currentData[j*dataRes+i];
    }
}

bool gpuMakeStep(GpuState* state, float startTime, float endTime) {

    cudaEvent_t start, stop,startT, stopT;
    cudaEventCreate(&start);
    cudaEventCreate(&startT);
    cudaEventCreate(&stop);
    cudaEventCreate(&stopT);

    cudaError_t cudaStatus;
    dim3 blocks;
    dim3 threads;

    cudaEventRecord(startT);



    cudaMemsetAsync(state->currentData, 0x0, sizeof(FieldComponent)*state->gridDim*12*state->workingBlocksCount,state->modellingStream);
    cudaStatus = cudaStreamSynchronize(state->modellingStream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemsetAsync failed! currentData");
        return NULL;
    }


    int x=0;
    float milliseconds = 0;
    do {
        x++;
        state->exchangeCounters[2] = 0;




        blocks.x = state->workingBlocksCount;
        blocks.y = 1;
        blocks.z = 1;
        threads.x = 1024;
        threads.y = 1;
        threads.z = 1;

        //cudaEventRecord(start);
        makeStep << < blocks, threads, 0, state->modellingStream >> >
                                          (state->particlesBlocks, state->particlesBlocksCount, state->electricData, state->magneticData, state->currentData, startTime, endTime, state->gridNx, state->gridNy, state->gridNz, state->gridData, state->exchangeCounters
                                          );
        cudaStatus = cudaStreamSynchronize(state->modellingStream);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Kernel failed: makeStep " << cudaStatus << "\n";
            return false;
        }

        //cudaEventRecord(stop);
        //cudaEventSynchronize(stop);
        //milliseconds = 0;
        //cudaEventElapsedTime(&milliseconds, start, stop);
        //std::cout << "MODELLING DONE IN " << milliseconds << "ms " << x << "iteration\n";

    } while (state->exchangeCounters[2] != 0);

    cudaEventRecord(stopT);
    cudaEventSynchronize(stopT);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startT, stopT);
    //std::cout << "TOTAL " << milliseconds << "ms " << x << "iterations\n";

    int blocksCount = state->gridDim*12/1024;
    if(state->gridDim*12%1024 != 0) {
        ++blocksCount;
    }

    blocks.x = blocksCount;
    blocks.y = 1;
    blocks.z = 1;
    threads.x = 1024;
    threads.y = 1;
    threads.z = 1;
    collectCurrentData<< < blocks, threads, 0, state->modellingStream >>>(state->currentData, 12*state->gridDim,state->workingBlocksCount);
    cudaStatus = cudaStreamSynchronize(state->modellingStream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel failed: makeStep " << cudaStatus << "\n";
        return false;
    }

    cudaMemcpyAsync(state->currentDataHost,state->currentData, sizeof(FieldComponent)*state->gridDim*12,cudaMemcpyDeviceToHost,state->modellingStream);
    cudaStatus = cudaStreamSynchronize(state->modellingStream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpyAsync failed: currentData " << cudaStatus << "\n";
        return false;
    }


    return true;
}

