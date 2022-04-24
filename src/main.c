
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#include "shared_memory.h"
#include "matrix.h"
#include "file_reader.h"
#include "determinant_calculation.h"
#include "constants.h"

/**
 *  \file main.c
 *
 *  \brief Program computing the matrices inside files
 *
 *  This program reads several text files whose names are provided in the command line and proceeds to compute the matrices inside each file
 *  
 *  To carry out this task 1 or more concurrent computing worker threads and reading working threads are launched.
 * 
 *  \author João Diogo Ferreira, João Tiago Rainho - April 2022
 */


/** \brief number of nano seconds in a second */
#define NS_PER_SECOND 1000000000

/** \brief producer threads return status array */
int statusProd[N_FILE_READER_WORKERS];

/** \brief consumer threads return status array */
int statusCons[N_DETERMINANT_WORKERS];

int startWorkers(int nDeterminantWorkers, int nReadingWorkers);

/** \brief Main thread.
 *  
 *  The role of main thread is to get file names by processing the command line and storing them.
*/
int main(int argc, char *argv[]) {

    int opt;

    char * fileNames[((argc-1)/2)+1];
    unsigned int nFiles = 0;

    do {
        switch((opt = getopt(argc, argv, "f:h"))) {
            case 'f':
                fileNames[nFiles] = optarg;
                nFiles++;
                break;
                
            case 'h':
                printf("-f      --- filename\n");
                break;
        }
    }
    while(opt != -1);

    sm_init(fileNames, nFiles);

    // initialize time variables and start clock
    struct timespec startTime, stopTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime);

    startWorkers(N_DETERMINANT_WORKERS, N_FILE_READER_WORKERS);

    // stop clock and compute elapsed time
    clock_gettime(CLOCK_MONOTONIC, &stopTime);

    // print results
    FileHandler * fileHandlers;
    sm_getResults(&fileHandlers);
    
    // print results
    for(int i=0;i<nFiles;i++) {
        printf("File: %s\nNumber of matrices to be read = %d\nOrder = %d\n", fileHandlers[i].fileName, fileHandlers[i].nMatrices, fileHandlers[i].order);
        for(int j=0;j<fileHandlers[i].nMatrices;j++) {
            printf("The determinant of matrix %d is %.3e\n", j, fileHandlers[i].determinants[j]);
        }
    }
    
    printf ("\nElapsed time = %.6f s\n",  (stopTime.tv_sec - startTime.tv_sec) / 1.0 + (stopTime.tv_nsec - startTime.tv_nsec) / 1000000000.0);
    
    // free memory
    sm_close();

    return EXIT_SUCCESS;
}

/** \brief startWorkers.
 * 
 * This function starts the worker threads and waits for their termination, when both the reading threads and the determinant computation threads finish their lifecycle, it prints the results.
 * 
 */
int startWorkers(int nDeterminantWorkers, int nReadingWorkers) {

    // thread initialization
    pthread_t computingThreadWorkers[nDeterminantWorkers];
    pthread_t readingThreadWorkers[nReadingWorkers];

    // thread Ids
    int computingThreadIds[nDeterminantWorkers];
    int readingThreadIds[nDeterminantWorkers];

    // create concurrent threads
    for(int i=0;i<nDeterminantWorkers;i++) {
        computingThreadIds[i] = i;
        pthread_create(&computingThreadWorkers[i], NULL, compute_determinant_thread_worker, (void*) &computingThreadIds[i]);
    }
    for(int i=0;i<nReadingWorkers;i++) {
        readingThreadIds[i] = i;
        pthread_create(&readingThreadWorkers[i], NULL, file_reader_thread_worker, (void*) &readingThreadIds[i]);
    }

    // end reading threads
    for(int i=0;i<nReadingWorkers;i++) {
        pthread_join(readingThreadWorkers[i], NULL);
        printf("Finished Reading Thread with id %d\n", i);
    }
    // signal the shared memory there is no more matrices to add
    doneReading();

    // finish computing threads
    for(int i=0;i<nDeterminantWorkers;i++) {
        pthread_join(computingThreadWorkers[i], NULL);
        printf("Finished Computing Thread with id %d\n", i);
    }

    return EXIT_SUCCESS;
}