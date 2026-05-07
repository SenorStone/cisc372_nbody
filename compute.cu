#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

static vector3 *dAccels = NULL;
static vector3 *dPos    = NULL;
static vector3 *dVel    = NULL;
static double  *dMass   = NULL;

__global__ void computeAccels(vector3 *accels, const vector3 *pos, const double *dmass, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;

    if (i == j) {
        accels[i*n+j][0] = accels[i*n+j][1] = accels[i*n+j][2] = 0.0;
        return;
    }

    double dx = pos[i][0] - pos[j][0];
    double dy = pos[i][1] - pos[j][1];
    double dz = pos[i][2] - pos[j][2];
    double mag_sq = dx*dx + dy*dy + dz*dz;
    double mag = sqrt(mag_sq);
    double amag = -GRAV_CONSTANT * dmass[j] / mag_sq;

    accels[i*n+j][0] = amag * dx / mag;
    accels[i*n+j][1] = amag * dy / mag;
    accels[i*n+j][2] = amag * dz / mag;
}
__global__ void updateBodies(const vector3 *accels, vector3 *pos, vector3 *vel, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int j = 0; j < n; j++) {
        ax += accels[i*n+j][0];
        ay += accels[i*n+j][1];
        az += accels[i*n+j][2];
    }

    vel[i][0] += ax * INTERVAL;
    vel[i][1] += ay * INTERVAL;
    vel[i][2] += az * INTERVAL;
    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}

void initDeviceMemory(void)
{
    int N = NUMENTITIES;
    cudaMalloc((void**)&dAccels, sizeof(vector3) * N * N);
    cudaMalloc((void**)&dPos, sizeof(vector3) * N);
    cudaMalloc((void**)&dVel, sizeof(vector3) * N);
    cudaMalloc((void**)&dMass, sizeof(double) * N);
	cudaMemcpy(dMass, mass, sizeof(double)*N, cudaMemcpyHostToDevice);
}
void freeDeviceMemory(void)
{
    cudaFree(dAccels);
    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
}

void compute(void)
{
    int N = NUMENTITIES;
    cudaMemcpy(dPos, hPos, sizeof(vector3)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, sizeof(vector3)*N, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((unsigned int)ceil((double)N/16.0), (unsigned int)ceil((double)N/16.0));
    computeAccels<<<grid, block>>>(dAccels, dPos, dMass, N);
    int threads = 256;
	int blocks = (int)ceil((double)N / threads);
	updateBodies<<<blocks, threads>>>(dAccels, dPos, dVel, N);

    cudaMemcpy(hPos, dPos, sizeof(vector3)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, sizeof(vector3)*N, cudaMemcpyDeviceToHost);
}
