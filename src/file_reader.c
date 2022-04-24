
#include <stdio.h>
#include <stdlib.h>

#include "shared_memory.h"
#include "fifo.h"

void * file_reader_thread_worker(void * arg) {
    unsigned int threadId = *((int*) arg);
    bool hasMoreWork = true;
    while(hasMoreWork) {
        FileHandler * fileHandler;
        unsigned int fileIdx;
        hasMoreWork = sm_getFile(&fileHandler, &fileIdx);

        if(hasMoreWork==true) {
            // open file
            FILE * ptrFile = fopen(fileHandler->fileName, "r");

            // check if file exists
            if(ptrFile == NULL) {
                perror("Error opening file");
                printf("%s\n", fileHandler->fileName);
                return (void *) EXIT_FAILURE;
            }

            
            unsigned int nMatrices;
            unsigned int order;

            // get number of matrices and the order of the matrices
            fread(&nMatrices, sizeof(unsigned int), 1, ptrFile);
            fread(&order, sizeof(unsigned int), 1, ptrFile);

            fileHandler->nMatrices = nMatrices;
            fileHandler->determinants = (double*) malloc(sizeof(double)*nMatrices);
            fileHandler->order = order;

            for(int i=0;i<nMatrices;i++) {

                MatrixHandler * matrixHandler = (MatrixHandler*) malloc(sizeof(MatrixHandler));
                matrixHandler->fileIdx = fileIdx;
                matrixHandler->matrixIdx = i;

                matrixHandler->matrix = (Matrix*) malloc(sizeof(Matrix));

                matrixHandler->matrix->numbers = (double **) malloc(sizeof(double*)*order);
                matrixHandler->matrix->order = order;

                for(int i=0;i<order;i++) {
                    // read matrix from files
                    matrixHandler->matrix->numbers[i] = (double*) malloc(sizeof(double)*order);
                    fread(matrixHandler->matrix->numbers[i], sizeof(double), order, ptrFile);
                }

                putMatrix(threadId, matrixHandler);
            }
        }
    }

    return EXIT_SUCCESS;

}