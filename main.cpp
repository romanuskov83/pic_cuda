#include <iostream>

#include <common.h>
#include <math.h>
#include <cstring>
#include "gpu_code.h"
#include "random.h"
int main() {
    const int nx = 100;
    const int ny = 100;
    const int nz = 100;
    const int gridDim = nx*ny*nz;
    initRandom(1);

    Coordinate gridX[nx+1];
    Coordinate gridY[ny+1];
    Coordinate gridZ[nz+1];
    for(int i = 0; i <= nx; i++) {
        gridX[i] = i*100;
    }

    for(int i = 0; i <= ny; i++) {
        gridY[i] = i*100;
    }

    for(int i = 0; i <= nz; i++) {
        gridZ[i] = i*100;
    }

    GpuState* state = gpuInit(0,nx,ny,nz,gridX,gridY,gridZ);

    FieldComponent* electricData = (FieldComponent *)malloc(sizeof(FieldComponent)*((nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + (nx+1)*(ny+1)*(nz)));
    memset(electricData,0x0,sizeof(FieldComponent)*((nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + (nx+1)*(ny+1)*(nz)));
    FieldComponent* magneticData = (FieldComponent *)malloc(sizeof(FieldComponent)*((nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz) + (nx)*(ny)*(nz+1)));
    memset(magneticData,0x0,sizeof(FieldComponent)*((nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz) + (nx)*(ny)*(nz+1)));
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nz+1; k++) {
                magneticData[(nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz)+cel_idx(i,j,k,nx,ny,nz+1)] = 0.5f;
            }
        }
    }

    for(int i = 0; i < nx+1; i++) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nz+1; k++) {
                //electricData[(nx)*(ny+1)*(nz+1) + cel_idx(i,j,k,nx+1,ny,nz+1)] = -0.1f;
            }
        }
    }


    gpuUpdateFieldsData(state,electricData,magneticData);

    int countIn;
    int countOut;

    float step = 1e-9f;
    float time = 0.0f;
    int outParticles = 0;

    int steps = 1000;
    int ADDITIONS_PER_STEP = 1;
    ParticleInfo *info = (ParticleInfo *) malloc(EXCHANGE_PARTICLES_COUNT * sizeof(ParticleInfo));
    int partPerCell = (EXCHANGE_PARTICLES_COUNT/gridDim);
    countIn = 1;
    for (int i = 0; i < countIn; i++) {
        int cellId = i % gridDim;
        //        int cellId = i / partPerCell;

        const int ix = nx/2;//x_idx(cellId,nx,ny,nz);
        const int iy = 0;y_idx(cellId,nx,ny,nz);
        const int iz = 0;//z_idx(cellId,nx,ny,nz);

        info[i].rx = gridX[ix] + getRandom(0) * (gridX[ix+1] - gridX[ix]);
        info[i].ry = gridY[iy] + getRandom(1) * (gridY[iy+1] - gridY[iy]);
        info[i].rz = gridZ[iz] + getRandom(2) * (gridZ[iz+1] - gridZ[iz]);


        float phi = 2 * 3.1415f * getRandom(3);
        float tetta = 3.1415f * getRandom(4);
        float energy = 0.01f;//1.4f; //meV
        float p = sqrtf(sqr((energy+E0e)/E0e)-1.0f);

        info[i].px = p;//sinf(tetta) * cosf(phi)*p;
        info[i].py = 0;//sinf(tetta) * sinf(phi)*p;
        info[i].pz = 0;//cosf(tetta) * p;

        info[i].currentTime = time;
        info[i].weight = 1;
    }

    do {
        gpuExchangeParticles(state, info, countIn, countOut);
        for (int i = 0; i < fmin(countOut, EXCHANGE_PARTICLES_COUNT); i++) {
            int ix = (info[countIn + i].cellIdFlag & CELL_ID_MASK) / (ny * nz);
            int iy = (info[countIn + i].cellIdFlag & CELL_ID_MASK) / (nz) % ny;
            int iz = (info[countIn + i].cellIdFlag & CELL_ID_MASK) % nz;
            int flag = info[countIn + i].cellIdFlag & FLAG_MASK;
            outParticles++;
            /*std::cout << "P out (" << info[countIn+i].rx << "\t" << info[countIn+i].ry << "\t" << info[countIn+i].rz << ")\t("
                    << info[countIn+i].px << "\t" << info[countIn+i].py << "\t" << info[countIn+i].pz << ")\t" << info[countIn+i].currentTime << "\t(" << ix << "\t" << iy << "\t" << iz << ")";
            if(flag == FLAG_FLEW_MINUS_X) {
                std::cout << " -X\n";
            }
            if(flag == FLAG_FLEW_PLUS_X) {
                std::cout << " +X\n";
            }
            if(flag == FLAG_FLEW_MINUS_Y) {
                std::cout << " -Y\n";
            }
            if(flag == FLAG_FLEW_PLUS_Y) {
                std::cout << " +Y\n";
            }
            if(flag == FLAG_FLEW_MINUS_Z) {
                std::cout << " -Z\n";
            }
            if(flag == FLAG_FLEW_PLUS_Z) {
                std::cout << " +Z\n";
            }*/
        }

    } while (countIn != 0 || countOut != 0);

    while(steps--) {

        gpuMakeStep(state, time, time+step);
        time += step;
        //std::cout << "TIME " << time << " OUT COUNT " << outParticles << "\n";
        /*for(int i = 0; i < nx; i++) {
            for(int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    printf("[%d %d %d]\n", i, j, k);
                    for(int l =0; l < 3; l++) {
                        int cellIdx = (i*state->gridNy + j)*state->gridNz+k;
                        printf("%d %f\n", l, state->currentDataHost[cellIdx*12+l]);
                    }

                }
            }
        }*/

    }

    return 0;
}