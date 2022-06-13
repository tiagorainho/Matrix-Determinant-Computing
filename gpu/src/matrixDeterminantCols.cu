#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>

// (row, col, order)
#define idx(x,y,order)(x*order+y)

#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KNRM  "\x1B[0m"

inline void switch_row(double *matrix, int row1, int row2, int order) {
    double aux;
    
    for(int i = 0; i < order; i++) 
    {
        aux = matrix[idx(row1,i,order)];
        matrix[idx(row1,i,order)] = matrix[idx(row2,i,order)];
        matrix[idx(row2,i,order)] = aux;
    }
}

void determinantOnHostColumns(double *matrices, int numberOfMatrix, double *determinant, int order)
{
    int sign = 1;
    double ratio = 1;
    
    for(int n = 0; n < numberOfMatrix; n++)
    {
        double *matrix = &matrices[n * order * order];
        determinant[n] = 1;
        
        // for each row
        for(int i = 0; i < order; i++) 
        {
            // check if the row can be used, otherwise, switch that row
            if(matrix[idx(i,i,order)] == 0) {
                bool determinantIsZero = true;
                for(int j = i+1; j < order; j++) 
                {
                    if(matrix[idx(j,i,order)] != 0) 
                    {
                        switch_row(matrix, i, j, order);
                        sign = (sign == 1) ? -1: 1;
                        determinantIsZero = false;
                        break;
                    }
                }
                if(determinantIsZero)
                {
                    determinant[n] = 0;
                    break;
                }                       
            }

            // for all other rows
            for(int j = i + 1; j < order; j++) {
                ratio = matrix[idx(j,i,order)] / matrix[idx(i,i,order)];
                for(int k = 0; k < order; k++) {
                    matrix[idx(j,k,order)] = matrix[idx(j,k,order)] - ratio * matrix[idx(i,k,order)];
                }
            }
            determinant[n] *= matrix[idx(i,i,order)];
        }
        determinant[n] *= sign;
    }
}

__global__ void determinantOnGPUColumns(double *mat, double *determinant, int order)
{
    extern __shared__ double tmp[];  

    unsigned int columnNumber = threadIdx.x + threadIdx.y * blockDim.x;
    unsigned int matrixNumber = blockIdx.x + blockIdx.y * gridDim.x;
    unsigned int size = order * order;

    double *matrix = &mat[matrixNumber * size];

    int sign = 1;
    determinant[matrixNumber] = 1;
    
    //for each row
    for(int i = 0; i < order; i++)
    {
        //swap row if necessary
        if(matrix[idx(i,i,order)] == 0) {
            bool determinantIsZero = true;
            for(int j= i + 1; j < order; j++) {
                if(matrix[idx(j,i,order)] != 0) {
                    double t = matrix[idx(i,columnNumber,order)];
                    matrix[idx(i,columnNumber,order)] = matrix[idx(j,columnNumber,order)];
                    matrix[idx(j,columnNumber,order)] = t;
                    sign = (sign == 1) ? -1 : 1;
                    determinantIsZero = false;

                    break;
                }
            } 
            //determinant is 0
            if(determinantIsZero)
            {            
                if(columnNumber == 0)
                    determinant[matrixNumber] = 0;
                return;               
            }
        }
        __syncthreads();

        //Read all necessary values
        tmp[columnNumber] = matrix[idx(columnNumber,i,order)];
        __syncthreads();

        //For all other rows
        for(int j = i + 1; j < order; j++)
        {
            double ratio = tmp[j] / matrix[idx(i,i,order)];

            //process corresponding col
            matrix[idx(j,columnNumber,order)] = matrix[idx(j,columnNumber,order)]-ratio*matrix[idx(i,columnNumber,order)];
        }
        //__syncthreads();
    }

    //calculate determinant
    if(columnNumber == 0)
    {
        determinant[matrixNumber] = sign;
        for(int i = 0; i < order; i++)
            determinant[matrixNumber] = determinant[matrixNumber] * matrix[idx(i,i,order)];
    }
}

void checkResult(double *cpuRef, double *gpuRef, int nDeterminants)
{
    bool match = 1;
    for(int i = 0; i < nDeterminants; i++)
    {
        double epsilon = (1 - cpuRef[i] / gpuRef[i]) * 100;
        if(epsilon < 0)
            epsilon = -epsilon;

        if (epsilon > 0.000001)
        {
            match = 0;
            printf("%sError: Matrix %3d - host %.8e \t gpu %.8e\n%s", KRED, i + 1, cpuRef[i], gpuRef[i], KNRM);
            break;
        }

        //printf("%sCorrect: Matrix %3d - host %.3e \t gpu %.3e\n%s", KGRN, i + 1, cpuRef[i], gpuRef[i], KNRM);
    }

    if (match)
        printf("Determinants match.\n\n");
    else
        printf("Determinants do not match.\n\n");
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // process cli 
    int opt;
    char * fileName;

    do {
        switch((opt = getopt(argc, argv, "f:h"))) {
            case 'f':
                fileName = optarg;
                break;
                
            case 'h':
                printf("-f      --- filename\n");
                break;
        }
    }
    while(opt != -1);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int order;
    int numberOfMatrix;
    
    FILE * ptrFile = fopen(fileName, "r"); 
    if(ptrFile == NULL) {
        fprintf(stderr,"Error opening file");
        printf("%s\n", fileName);
        exit(EXIT_FAILURE);
    }     
    size_t size = fread(&numberOfMatrix, sizeof(unsigned int), 1, ptrFile);
    if(size != 1)
    {
        fprintf(stderr,"Error reading the number of matrix in the file\n");
        exit(EXIT_FAILURE);
    }

    size = fread(&order, sizeof(unsigned int), 1, ptrFile);
    if(size != 1)
    {
        fprintf(stderr,"Error reading order from file\n");
        exit(EXIT_FAILURE);
    }

    printf("Filename: %s\nNumber of matrices: %d\nMatrices order: %d\n", fileName, numberOfMatrix, order);


    //host memory
    double *h_matrices = (double *)malloc(order*order*numberOfMatrix*sizeof(double));
    size = fread(h_matrices, sizeof(double), order*order*numberOfMatrix, ptrFile);
    if(size != order*order*numberOfMatrix)
    {
        fprintf(stderr,"Error matrices from file\n");
        exit(EXIT_FAILURE);
    }

    double determinantRefCPU[numberOfMatrix];
    double determinantRefGPU[numberOfMatrix];

    int nBytesMatrices = order * order * numberOfMatrix * sizeof(double);
    int nBytesDeterminants = numberOfMatrix * sizeof(double);


    // malloc device global memory
    double *d_matrices;
    double *d_determinant;

    CHECK(cudaMalloc((void **)&d_matrices, nBytesMatrices));
    CHECK(cudaMalloc((void **)&d_determinant, nBytesDeterminants));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_matrices, h_matrices, nBytesMatrices, cudaMemcpyHostToDevice));

    dim3 block(order, 1);
    dim3 grid(numberOfMatrix, 1);

    double iStart;
    double iElaps;
    
    // calculate determinant at host side
    iStart = seconds();
    determinantOnHostColumns(h_matrices, numberOfMatrix, determinantRefCPU, order);
    iElaps = seconds() - iStart;
    printf("determinantOnHostColumns elapsed %f sec\n", iElaps);

    // invoke kernel at host side
    iStart = seconds();
    determinantOnGPUColumns<<<grid, block, order * sizeof(double)>>>(d_matrices, d_determinant, order);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("determinantOnGPUColumns <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x,
           grid.y,
           block.x, block.y, iElaps);


    // check kernel error
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    CHECK(cudaMemcpy(determinantRefGPU, d_determinant, nBytesDeterminants, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(determinantRefCPU, determinantRefGPU, numberOfMatrix);

    // free device global memory
    CHECK(cudaFree(d_matrices));
    CHECK(cudaFree(d_determinant));

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
