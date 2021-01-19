#ifndef  __COMMON_H__PIC__
#define __COMMON_H__PIC__

typedef float Coordinate;
typedef float FieldComponent;
#define PARTICLE_BLOCK_SIZE 64

#define FLAG_DIED 0
#define FLAG_OK 0x20000000
#define FLAG_FLEW_PLUS_X 0x40000000
#define FLAG_FLEW_MINUS_X 0x60000000
#define FLAG_FLEW_PLUS_Y 0x80000000
#define FLAG_FLEW_MINUS_Y 0xA0000000
#define FLAG_FLEW_PLUS_Z 0xC0000000
#define FLAG_FLEW_MINUS_Z 0xE0000000

#define FLAG_MASK 0xE0000000
#define CELL_ID_MASK 0x1FFFFFFF

#define C 2.99792458e10f
#define Qe 4.80325021e-10f
#define Ee (-1)
#define Me 9.10953447e-28f
#define W2MeV 1.6021766e-6f

//Энергия покоя электрона в МэВ
#define E0e (Me*C*C/W2MeV)

#define sqr(x) ((x)*(x))


#define x_idx(cellIdx,gridNx,gridNy,gridNz) ((cellIdx)/((gridNz)*(gridNy)))
#define y_idx(cellIdx,gridNx,gridNy,gridNz) (((cellIdx)/(gridNz))%(gridNy))
#define z_idx(cellIdx,gridNx,gridNy,gridNz) ((cellIdx)%(gridNz))
#define cel_idx(iX,iY,iZ,nX,nY,nZ) (((iX)*(nY) + (iY))*(nZ)+(iZ))


typedef struct {
    Coordinate rx[PARTICLE_BLOCK_SIZE];
    Coordinate ry[PARTICLE_BLOCK_SIZE];
    Coordinate rz[PARTICLE_BLOCK_SIZE];
    Coordinate px[PARTICLE_BLOCK_SIZE];
    Coordinate py[PARTICLE_BLOCK_SIZE];
    Coordinate pz[PARTICLE_BLOCK_SIZE];
    float weight[PARTICLE_BLOCK_SIZE];
    float currentTime[PARTICLE_BLOCK_SIZE];
    unsigned int cellIdFlag[PARTICLE_BLOCK_SIZE];
    long id[PARTICLE_BLOCK_SIZE];
} ParticlesBlock;

typedef struct {
    Coordinate rx;
    Coordinate ry;
    Coordinate rz;
    Coordinate px;
    Coordinate py;
    Coordinate pz;
    float weight;
    float currentTime;
    int cellIdFlag;
    long id;
} ParticleInfo;

#endif //__COMMON_H__PIC__
