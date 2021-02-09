#include <iostream>

#include <common.h>
#include <math.h>
#include <cstring>
#include <fstream>
#include <csignal>
#include "gpu_code.h"
#include "random.h"

int steps = 5000000;


void my_handler(int s){
    steps = 0;
}


int main(int argc, char* argv[]) {
    const int nx = 501;
    const int ny = 501;
    const int nz = 501;
    const int gridDim = nx*ny*nz;
    initRandom(1);
    signal (SIGINT,my_handler);

    Coordinate gridX[nx+1];
    Coordinate gridY[ny+1];
    Coordinate gridZ[nz+1];

    float cellSize = 400;

    for(int i = 0; i <= nx; i++) {
        gridX[i] = -nx*cellSize/2 + i*cellSize;
    }

    for(int i = 0; i <= ny; i++) {
        gridY[i] = -ny*cellSize/2 + i*cellSize;
    }

    for(int i = 0; i <= nz; i++) {
        gridZ[i] = -nz*cellSize/2 + i*cellSize;
    }

    GpuState* state = gpuInit(0,nx,ny,nz,gridX,gridY,gridZ);

    FieldComponent* electricData = (FieldComponent *)malloc(sizeof(FieldComponent)*((nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + (nx+1)*(ny+1)*(nz)));
    memset(electricData,0x0,sizeof(FieldComponent)*((nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + (nx+1)*(ny+1)*(nz)));


    FieldComponent* magneticData = (FieldComponent *)malloc(sizeof(FieldComponent)*((nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz) + (nx)*(ny)*(nz+1)));
    memset(magneticData,0x0,sizeof(FieldComponent)*((nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz) + (nx)*(ny)*(nz+1)));


    Coordinate rx;
    Coordinate ry;
    Coordinate rz;
    FieldComponent yhx;
    FieldComponent yhy;
    FieldComponent yhz;

    //X-component of H
    for(int i = 0; i < nx+1; i++) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nz; k++) {

                rx = (gridX[i]);
                ry = (gridY[j]+gridY[j+1])/2;
                rz = (gridZ[k]+gridZ[k+1])/2;

                float r = sqrtf(sqr(rx)+sqr(ry)+sqr(rz));
                if (r < A) {
                    yhx = 0;
                } else {
                    yhx = -3*H0*rx*rz*A*A*A/(2*powf(r,5));
                }

                magneticData[cel_idx(i,j,k,nx+1,ny,nz)] = yhx;
            }
        }
    }

    //Y-component of H
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny+1; j++) {
            for(int k = 0; k < nz; k++) {

                rx = (gridX[i]+gridX[i+1])/2;
                ry = (gridY[j]);
                rz = (gridZ[k]+gridZ[k+1])/2;


                float r = sqrtf(sqr(rx)+sqr(ry)+sqr(rz));
                if (r < A) {
                    yhy = 0;
                } else {
                    yhy = -3*H0*ry*rz*A*A*A/(2*powf(r,5));
                }

                magneticData[(nx+1)*(ny)*(nz) + cel_idx(i,j,k,nx,ny+1,nz)] = yhy;
            }
        }
    }

    //Z-component of H
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny; j++) {
            for(int k = 0; k < nz+1; k++) {
                rx = (gridX[i]+gridX[i+1])/2;
                ry = (gridY[j]+gridY[j+1])/2;
                rz = (gridZ[k]);

                float r = sqrtf(sqr(rx)+sqr(ry)+sqr(rz));

                if (r < A) {
                    yhz = 0;
                } else {
                    yhz = H0*(1+A*A*A*((rx*rx+ry*ry)/2-rz*rz)/powf(r,5));
                }

                magneticData[(nx+1)*(ny)*(nz) + (nx)*(ny+1)*(nz)+cel_idx(i,j,k,nx,ny,nz+1)] = yhz;
            }
        }
    }


    //X-component of E
    for(int i = 0; i < nx; i++) {
        for(int j = 0; j < ny+1; j++) {
            for(int k = 0; k < nz+1; k++) {
                //electricData[cel_idx(i,j,k,nx,ny+1,nz+1)] = 0.01f;
            }
        }
    }

    //Z-component of E
    for(int i = 0; i < nx+1; i++) {
        for(int j = 0; j < ny+1; j++) {
            for(int k = 0; k < nz; k++) {
                //electricData[(nx)*(ny+1)*(nz+1) + (nx+1)*(ny)*(nz+1) + cel_idx(i,j,k,nx+1,ny+1,nz)] = 0.03f;
            }
        }
    }



    gpuUpdateFieldsData(state,electricData,magneticData);
    long id = 1;
    int countIn;
    int countOut;

    float step = argc > 2 ? atof(argv[2]) : 1e-9;
    float time = 0.0f;
    int outParticles = 0;


    ParticleInfo *info = (ParticleInfo *) malloc(EXCHANGE_PARTICLES_COUNT * sizeof(ParticleInfo));
    if(argc > 1) {
        int SOURCE_STEPS = 1;
        for(int a = 0; a < SOURCE_STEPS; a++) {
            countIn = 1;
            for(int x = 0; x < countIn; x++) {
                int i = 0;
                info[i].rx = 0;
                info[i].ry = 0;
                info[i].rz = 0;

                float degrees = atoi(argv[1]);

                float energy = 1.4f; //meV
                float p = sqrtf(sqr((energy + E0e) / E0e) - 1.0f);
                float phi = 0;//2*M_PI*getRandom(1);
                float tetta = M_PI*degrees/180;
                info[i].px = p*cosf(phi)*sinf(tetta);
                info[i].py = p*sinf(phi)*sinf(tetta);
                info[i].pz = p*cosf(tetta);

                info[i].currentTime = a*1e-5/SOURCE_STEPS;
                info[i].weight = 1;
                info[i].id = id;
                id++;
            }
            gpuExchangeParticles(state, info, countIn, countOut);
            if (countIn != 0) {
                std::cout << " WARNING: countIn != 0\n";
            }
        }
    } else {
        std::ifstream ifs("dumped.dat", std::ios::binary);

        int totalCount;
        int dummy;
        ifs.read(reinterpret_cast<char *>(&totalCount),sizeof(totalCount));
        ifs.read(reinterpret_cast<char *>(&time),sizeof(time));

        int i =0;
        for(int x =0; x < totalCount;x++) {
            ifs.read(reinterpret_cast<char *>(&info[i]),sizeof(info[i]));
            i++;
            if(i == EXCHANGE_PARTICLES_COUNT) {
                gpuExchangeParticles(state, info, i, dummy);
                if (i != 0) {
                    std::cout << " WARNING: countIn2 != 0\n";
                    i = 0;
                }

            }
        }

        if(i != 0) {
            gpuExchangeParticles(state, info, i, dummy);
            if (i != 0) {
                std::cout << " WARNING: countIn2 != 0\n";
            }
        }


        ifs.close();
    }

    int totalOut = 0;
    while(steps-- > 0) {

        gpuMakeStep(state, time, time+step);
        time += step;
        if(steps % 100 == 0) {
            countIn = 0;
            gpuExchangeParticles(state, info, countIn, countOut);
            totalOut += countOut;
            if(countOut > 0) {
                //std::cout << " " << time << " " << totalOut << " " <<info[0].id << "\n";
                exit(0);
            } else {
                //std::cout << " " << time << " " << totalOut << " " << -1 << "\n";
            }

        }
    }

    std::ofstream ofs("dumped.dat", std::ios::binary);

    bool isFirst = true;

    while(1) {
        gpuDumpParticles(state, info, countOut);
        if(countOut == 0)
            break;

        if(isFirst)
        {
            ofs.write(reinterpret_cast<char *>(&countOut),sizeof(countOut));
            ofs.write(reinterpret_cast<char *>(&time),sizeof(time));
            isFirst = false;
        }

        for(int i = 0; i < countOut && i < EXCHANGE_PARTICLES_COUNT; i++) {
            ofs.write(reinterpret_cast<char *>(&info[i]),sizeof(info[i]));
        }
    }
    ofs.close();


    return 0;
}