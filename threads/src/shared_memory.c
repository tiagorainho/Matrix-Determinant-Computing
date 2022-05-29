
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#include "shared_memory.h"


/** \brief mutex used for the threads syncronization */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/** \brief Number of files provided in the initialization */
static unsigned int nFiles;

/** \brief Array of FileHandlers */
static FileHandler* files;

/** \brief integer used to represent in which file the processing was the last that took place */
static unsigned int currentFileIdx;


void sm_getResults(FileHandler ** results) {
    *results = files;
}


void sm_init(char ** fileNames, unsigned int numberFiles) {
    // variable initialization
    nFiles = numberFiles;
    currentFileIdx = 0;
    files = (FileHandler*) malloc(nFiles * sizeof(FileHandler));
    
    for(int i=0;i<nFiles;i++) {
        files[i].fileName = (char*) malloc(strlen(fileNames[i])*sizeof(char));
        strcpy(files[i].fileName, fileNames[i]);
    }
}

void sm_close() {
    // delete file handlers
    for(int i=0;i<nFiles;i++) {
        free(files[i].fileName);
        free(files[i].determinants);
    }
    free(files);
}


void sm_registerResult(MatrixHandler * matrixHandler, double result) {
    FileHandler fh = files[matrixHandler->fileIdx];
    fh.determinants[matrixHandler->matrixIdx] = result;

    // free matrix handler
    for(int i=0;i<matrixHandler->matrix->order;i++) {
        free(matrixHandler->matrix->numbers[i]);
    }
    free(matrixHandler->matrix->numbers);
    free(matrixHandler->matrix);
    free(matrixHandler);
}

bool sm_getMatrix(MatrixHandler * matrixHandler) {
    unsigned int threadId = 0;
    return getMatrix(threadId, &matrixHandler);
}

void sm_addMatrix(MatrixHandler matrixHandler) {
    unsigned int threadId = 0;
    putMatrix(threadId, &matrixHandler);
}

bool sm_getFile(FileHandler ** fileHandler, unsigned int * fileIdx) {
    pthread_mutex_lock(&mutex);
    bool hasMoreWork = false;

    if(currentFileIdx < nFiles) {
        *fileHandler = &files[currentFileIdx];
        *fileIdx = currentFileIdx;
        currentFileIdx++;
        hasMoreWork = true;
    }

    pthread_mutex_unlock(&mutex);
    return hasMoreWork;
}