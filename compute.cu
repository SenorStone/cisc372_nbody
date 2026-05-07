#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

static vector3 *dAccels = NULL;
static vector3 *dPos    = NULL;
static vector3 *dVel    = NULL;
static double  *dMass   = NULL;

// Compute accelerations and update positions/velocities on the GPU
__global__ void computeAccels(vector3 *accels, const vector3 *pos, const double *dmass, int n)
{
	__shared__ vector3 sPos[16];
    __shared__ double  sMass[16];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

	// Load positions and masses into shared memory for the current block
    if (threadIdx.y == 0 && j < n) {
        sPos[threadIdx.x][0] = pos[j][0];
        sPos[threadIdx.x][1] = pos[j][1];
        sPos[threadIdx.x][2] = pos[j][2];
        sMass[threadIdx.x]   = dmass[j];
    }
    __syncthreads();

    if (i >= n || j >= n) return;

    if (i == j) {
        accels[i*n+j][0] = accels[i*n+j][1] = accels[i*n+j][2] = 0.0;
        return;
    }
	// Compute acceleration contribution from body j to body i
    double dx = pos[i][0] - sPos[threadIdx.x][0];
    double dy = pos[i][1] - sPos[threadIdx.x][1];
    double dz = pos[i][2] - sPos[threadIdx.x][2];
    double mag_sq = dx*dx + dy*dy + dz*dz;
    double mag = sqrt(mag_sq);
    double amag = -GRAV_CONSTANT * sMass[threadIdx.x] / mag_sq;

    accels[i*n+j][0] = amag * dx / mag;
    accels[i*n+j][1] = amag * dy / mag;
    accels[i*n+j][2] = amag * dz / mag;
}
// Update velocities and positions based on computed accelerations
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
	// Update velocity and position
    vel[i][0] += ax * INTERVAL;
    vel[i][1] += ay * INTERVAL;
    vel[i][2] += az * INTERVAL;
    pos[i][0] += vel[i][0] * INTERVAL;
    pos[i][1] += vel[i][1] * INTERVAL;
    pos[i][2] += vel[i][2] * INTERVAL;
}
// Initialize device memory for positions, velocities, masses, and accelerations
extern "C" void initDeviceMemory(void)
{
    int N = NUMENTITIES;
    cudaMalloc((void**)&dAccels, sizeof(vector3) * N * N);
    cudaMalloc((void**)&dPos, sizeof(vector3) * N);
    cudaMalloc((void**)&dVel, sizeof(vector3) * N);
    cudaMalloc((void**)&dMass, sizeof(double) * N);
	cudaMemcpy(dMass, mass, sizeof(double)*N, cudaMemcpyHostToDevice);
}
// Free device memory after computation is done
extern "C" void freeDeviceMemory(void)
{
    cudaFree(dAccels);
    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
}

extern "C" void compute(void)
{
	// Copy positions and velocities to the device
    int N = NUMENTITIES;
    cudaMemcpy(dPos, hPos, sizeof(vector3)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dVel, hVel, sizeof(vector3)*N, cudaMemcpyHostToDevice);
	// Launch kernels to compute accelerations and update positions/velocities
    dim3 block(16, 16);
    dim3 grid((unsigned int)ceil((double)N/16.0), (unsigned int)ceil((double)N/16.0));
    computeAccels<<<grid, block>>>(dAccels, dPos, dMass, N);
    int threads = 256;
	int blocks = (int)ceil((double)N / threads);
	updateBodies<<<blocks, threads>>>(dAccels, dPos, dVel, N);
	// Copy updated positions and velocities back to the host
    cudaMemcpy(hPos, dPos, sizeof(vector3)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, sizeof(vector3)*N, cudaMemcpyDeviceToHost);
}
