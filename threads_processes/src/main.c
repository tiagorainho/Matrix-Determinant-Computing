
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <mpi.h>

#include "shared_memory.h"
#include "matrix.h"
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


/** \brief Reading threads return status */
int * statusReadingThread;

/** \brief Proxy threads return status */
int * statusProxyThread;

/** \brief number of nano seconds in a second */
#define NS_PER_SECOND 1000000000
#define VERBOSE 0

/** \brief producer threads return status array */
int statusProd[N_FILE_READER_WORKERS];

/** \brief consumer threads return status array */
int statusCons[N_DETERMINANT_WORKERS];

/** \brief function to dispatch all threads */
void dispatcher(int nWorkers, int * status);

/** \brief function which makes the determinant calculation */
void worker(int rank, int * status);

/** \brief thread function which reads files and adds the matrices inside a thread safe fifo */
void * file_reader_thread_worker(void * arg);

/** \brief thread function which pops matrices from a fifo and sends them to a worker */
void * proxyComputingThread(void * arg);

/** \brief function which terminants n workers */
void terminateWorkers(unsigned int nWorkers);

/** \brief Main thread.
 *  
 *  The role of main thread is to get file names by processing the command line and storing them.
*/
int main(int argc, char *argv[]) {

    // Determine executing start time
    struct timespec startTimeInit, endTimeInit;
    clock_gettime(CLOCK_MONOTONIC, &startTimeInit);
    
    // initialize MPI variables
    int rank, nProc, nWorkers, provided, workStatus;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("Warning MPI did not provide MPI_THREAD_FUNNELED\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    nWorkers = nProc - 1;

    // guarantee there is at least 1 worker process
    if (nWorkers < 1)
    {
        if (rank == 0)
            if(VERBOSE) printf("Wrong number of processes! It must be greater than 1.\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    if(VERBOSE) printf("Rank %d started\n", rank);

    if (rank == 0)
    {
        // process cli 
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

        // initialize monitor
        sm_init(fileNames, nFiles);

        // allocate thread status resources
        statusReadingThread = (int *) malloc(nWorkers * sizeof(int));
        statusProxyThread = (int *) malloc(nWorkers * sizeof(int));

        // Determine initialization time
        clock_gettime(CLOCK_MONOTONIC, &endTimeInit);
        printf("\nInitialization time = %.6f s\n", (endTimeInit.tv_sec - startTimeInit.tv_sec) / 1.0 + (endTimeInit.tv_nsec - startTimeInit.tv_nsec) / 1000000000.0);

        // initialize time variables and start clock
        struct timespec startTime, stopTime;
        clock_gettime(CLOCK_MONOTONIC, &startTime);

        // start dispatcher
        dispatcher(nWorkers, &workStatus);

        if(workStatus == EXIT_FAILURE) {
            printf("\nAn error has occured on dispatcher\n");
            MPI_Finalize();
            exit(0);
        }

        // stop time
        clock_gettime(CLOCK_MONOTONIC, &stopTime);

        // print results
        FileHandler * fileHandlers;
        sm_getResults(&fileHandlers);
        for(int i=0;i<nFiles;i++) {
            printf("File: %s\nNumber of matrices to be read = %d\nOrder = %d\n", fileHandlers[i].fileName, fileHandlers[i].nMatrices, fileHandlers[i].order);
            for(int j=0;j<fileHandlers[i].nMatrices;j++) {
                printf("The determinant of matrix %d is %.3e\n", j+1, fileHandlers[i].determinants[j]);
            }
        }

        // print time performance
        printf ("\nElapsed time = %.6f s\n",  (stopTime.tv_sec - startTime.tv_sec) / 1.0 + (stopTime.tv_nsec - startTime.tv_nsec) / 1000000000.0);

        // free memory
        sm_close();
    }
    else
    {
        worker(rank, &workStatus);
        if(workStatus == EXIT_FAILURE) {
            printf("\nAn error has occured on worker %d\n", rank);
            MPI_Finalize();
            exit(0);
        }
    }

    MPI_Finalize();
}

void dispatcher(int nWorkers, int * status) {

    // initialize variables used for processing and control
    int ready[nWorkers];
    int * executionStatus;
    bool success = true;
    MatrixHandler * matrixHandlers[nWorkers];
    MPI_Request recvRequests[nWorkers];
    MPI_Request sendRequests[nWorkers];
    double results[nWorkers];

    // create reading threads
    unsigned int nReadingWorkers = nWorkers;
    if(VERBOSE) printf("Start %d Reading workers\n", nReadingWorkers);
    pthread_t readingThreadWorkers[nReadingWorkers];
    int readingThreadIds[nReadingWorkers];

    // start reading threads
    for(int i=0;i<nReadingWorkers;i++) {
        readingThreadIds[i] = i;
        if(VERBOSE) printf("Start reading worker: %d\n", readingThreadIds[i]);
        if(pthread_create(&readingThreadWorkers[i], NULL, file_reader_thread_worker, (void*) &readingThreadIds[i]) != 0) {
            printf("Error on creating reading worker %d\n", readingThreadIds[i]);
            terminateWorkers(nWorkers);
            doneReading();
            *status = EXIT_FAILURE;
            return;
        }
    }
    
    // launch working threads
    pthread_t computingThreadWorkers[nWorkers];
    int computingThreadIds[nWorkers];
    for(int i=0;i<nWorkers;i++) {
        computingThreadIds[i] = i;
        if(VERBOSE) printf("Start proxy computing worker: %d\n", computingThreadIds[i]);
        if(pthread_create(&computingThreadWorkers[i], NULL, proxyComputingThread, (void*) &computingThreadIds[i]) != 0) {
            printf("Error on creating proxy computing worker %d\n", computingThreadIds[i]);
            terminateWorkers(nWorkers);
            doneReading();
            *status = EXIT_FAILURE;
            return;
        }
    }

    // end reading threads
    for(int i=0;i<nReadingWorkers;i++) {
        if(pthread_join(readingThreadWorkers[i], (void *) &executionStatus) != 0) {
            printf("Error on joining reader worker %d\n", readingThreadIds[i]);
            terminateWorkers(nWorkers);
            doneReading();
            *status = EXIT_FAILURE;
            return;
        }
        if(*executionStatus != 0) {
            printf("Error on joining middle reader worker %d\n", readingThreadIds[i]);
            doneReading();
            *status = EXIT_FAILURE;
            return;
        }
        if(VERBOSE) printf("Finished Reading Thread with id %d\n", readingThreadIds[i]);
    }
    
    // signal the shared memory there is no more matrices to add
    doneReading();

    // join working threads
    for(int i=0;i<nWorkers;i++) {
        if(pthread_join(computingThreadWorkers[i], (void *) &executionStatus) != 0) {
            printf("Error on proxy computing worker %d\n", readingThreadIds[i]);
            *status = EXIT_FAILURE;
            return;
        }

        if((*executionStatus) != 0) {
            *status = EXIT_FAILURE;
            return;
        }
        if(VERBOSE) printf("Finished computing Thread with id %d\n", computingThreadIds[i]+1);
    }
    *status = EXIT_SUCCESS;
}

void worker(int rank, int * status) {
    double result;
    int order;

    while (true)
    {
        // get order of the matrix
        MPI_Recv( (void *) &order, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Check is there is more work to do
        if(order == 0) {
            break;
        }

        if(VERBOSE) printf("Rank %d received order %d\n", rank, order);

        // receive actual matrix
        double numbers[order][order];
        MPI_Recv( (void *) numbers, order*order, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if(VERBOSE) printf("Rank %d received matrix starting with %f\n", rank, numbers[0][0]);

        // compute matrix
        Matrix * matrix = init_matrix(order, numbers);
        result = compute_determinant(*matrix);

        // return result
        MPI_Send( (void *) &result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        if(VERBOSE) printf("Rank %d return determinant %f\n", rank, result);
    }
    *status = EXIT_SUCCESS;
}

void * proxyComputingThread(void * arg) {
    unsigned int threadId = *((int*) arg);
    int processId = threadId + 1;
    MatrixHandler * matrixHandler;
    MPI_Status mpi_status;
    MPI_Request req_status;
    double result;
    int order;


    while(true) {
        // fetch matrix from the fifo
        int status = getMatrix(0, &matrixHandler);

        if(VERBOSE) printf("Rank 0: matrix status %d\n", status);

        if(!status) {
            if(VERBOSE) printf("Rank 0: no more work for process %d\n", processId);
            break;
        }

        order = matrixHandler->matrix->order;

        // populate numbers array
        double numbers[order][order];
        for(int j=0;j<order;j++)
            for(int k=0;k<order;k++)
                numbers[j][k] = matrixHandler->matrix->numbers[j][k];

        if(VERBOSE) printf("Rank 0: send order %d to process %d\n", order, processId);
        MPI_Send((void *) &order, 1, MPI_INT, processId, 0, MPI_COMM_WORLD);

        if(VERBOSE) printf("Rank 0: send matrix to process %d\n", processId);
        MPI_Send((void *) numbers, order*order, MPI_DOUBLE, processId, 0, MPI_COMM_WORLD);

        MPI_Recv((void *) &result, 1, MPI_DOUBLE, processId, 0, MPI_COMM_WORLD, &mpi_status);

        if(VERBOSE) printf("Rank 0: Received value %f from process %d\n", result, processId);
        sm_registerResult(matrixHandler, result);
    }

    // send message to worker to shutdown
    order = 0;
    MPI_Send((void *) &order, 1, MPI_INT, processId, 0, MPI_COMM_WORLD);

    statusProxyThread[threadId] = EXIT_SUCCESS;
    pthread_exit(&statusProxyThread[threadId]);
}


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
                statusReadingThread[threadId] = EXIT_FAILURE;
                pthread_exit(&statusReadingThread[threadId]);
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
    
    statusReadingThread[threadId] = EXIT_SUCCESS;
    pthread_exit(&statusReadingThread[threadId]);
}

void terminateWorkers(unsigned int nWorkers) {
    int order = 0;
    for(int i=1;i<=nWorkers;i++) {
        MPI_Send((void *) &order, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
}