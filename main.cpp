#include <iostream>

#include <common.h>
#include <math.h>
#include <cstring>
#include "gpu_code.h"
#include "random.h"
int main() {
    const int nx = 100;
    const int ny = 100;
    const int nz = 1;
    const int gridDim = nx*ny*nz;
    initRandom(1);

    Coordinate gridX[nx+1];
    Coordinate gridY[ny+1];
    Coordinate gridZ[nz+1];
    for(int i = 0; i <= nx; i++) {
        gridX[i] = i*600;
    }

    for(int i = 0; i <= ny; i++) {
        gridY[i] = i*600;
    }

    for(int i = 0; i <= nz; i++) {
        gridZ[i] = i*60000;
    }

    GpuState* state = gpuInit(0,nx,ny,nz,gridX,gridY,gridZ);

    FieldComponent* electricData = (FieldComponent *)malloc(sizeof(FieldComponent)*((nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + (nx+1)*(ny+1)*(nz)));
    memset(electricData,0x0,sizeof(FieldComponent)*((nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + (nx+1)*(ny+1)*(nz)));
    FieldComponent* magneticData = (FieldComponent *)malloc(sizeof(FieldComponent)*((nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz) + (nx)*(ny)*(nz+1)));
    memset(magneticData,0x0,sizeof(FieldComponent)*((nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz) + (nx)*(ny)*(nz+1)));

    //Y-component of H
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny+1; j++) {
            for(int k = 0; k < nz; k++) {
                //magneticData[(nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz)+cel_idx(i,j,k,nx,ny,nz+1)] = 0.5f;
                magneticData[(nx+1)*(ny)*(nz) +cel_idx(i,j,k,nx,ny+1,nz)] = 0.5f;
            }
        }
    }


    //X-component of E
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny+1; j++) {
            for(int k = 0; k < nz+1; k++) {
                electricData[cel_idx(i,j,k,nx,ny+1,nz+1)] = 0.01f;
            }
        }
    }

    //Z-component of E
    for(int i = 0; i < nx+1; i++) {
        for(int j = 0; j < ny+1; j++) {
            for(int k = 0; k < nz; k++) {
                electricData[(nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + cel_idx(i,j,k,nx+1,ny+1,nz)] = 0.03f;
            }
        }
    }



    gpuUpdateFieldsData(state,electricData,magneticData);
    long id = 1;
    int countIn;
    int countOut;

    float step = 1e-9f;
    float time = 0.0f;
    int outParticles = 0;

    int steps = 1000000;
    int ADDITIONS_PER_STEP = 16;
    ParticleInfo *info = (ParticleInfo *) malloc(EXCHANGE_PARTICLES_COUNT * sizeof(ParticleInfo));

    int totalCount = EXCHANGE_PARTICLES_COUNT*ADDITIONS_PER_STEP;

    for(int a = 0; a < ADDITIONS_PER_STEP; a++) {
        countIn = EXCHANGE_PARTICLES_COUNT;
        for (int i = 0; i < countIn; i++) {
            info[i].rx = (gridX[0]+gridX[nx])/2;
            info[i].ry = (gridY[0]+gridY[ny])/2;
            info[i].rz = (gridZ[0]+gridZ[nz])/2;


            //float phi = 2 * 3.1415f * getRandom(3);
            //float tetta = 3.1415f * getRandom(4);
            float nx = 1;
            float ny = 0;
            float nz = 3;
            float n = sqrtf(sqr(nx)+sqr(ny)+sqr(nz));
            nx /= n;
            ny /= n;
            nz /= n;


            float energy = 1.0f;//1.4f; //meV
            float p = sqrtf(sqr((energy + E0e) / E0e) - 1.0f);

            info[i].px = nx * p;
            info[i].py = ny * p;
            info[i].pz = nz * p;

            info[i].currentTime = id*8.530058e-06f/totalCount;
            info[i].weight = 1;
            info[i].id = id;
            id++;

        }
        gpuExchangeParticles(state, info, countIn, countOut);
        if (countIn != 0) {
            std::cout << " WARNING: countIn != 0\n";
        }
    }

    while(steps--) {

        gpuMakeStep(state, time, time+step);
        time += step;
        if(steps % 100 == 0) {
            countIn = 0;
            gpuExchangeParticles(state, info, countIn, countOut);
            for (int i = 0; i < fmin(countOut, EXCHANGE_PARTICLES_COUNT); i++) {
                printf("ID OUT %ld (%e,%e,%e) -> (%e, %e, %e) %e %e\n",info[countIn + i].id,info[countIn + i].rx,info[countIn + i].ry,info[countIn + i].rz,info[countIn + i].px,info[countIn + i].py,info[countIn + i].pz,info[countIn + i].currentTime, time);
                exit(0);
            }
            //std::cout << " Time is " << time << "\n";
        }
    }

    return 0;
}