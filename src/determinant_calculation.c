
#include <stdbool.h>
#include <stdlib.h>

#include "fifo.h"
#include "matrix.h"
#include "shared_memory.h"


void * compute_determinant_thread_worker(void * arg) {
    unsigned int threadId = *((int*) arg);
    bool continue_working = true;
    do {
        // initializations
        double determinant = 0;
        MatrixHandler * matrixHandler;

        // fetch matrix from the fifo
        continue_working = getMatrix(threadId, &matrixHandler);

        if(continue_working) {        
            // compute determinant
            determinant = compute_determinant(*(matrixHandler->matrix));

            // register
            sm_registerResult(matrixHandler, determinant);
        }
    } while(continue_working);

    return EXIT_SUCCESS;
}


