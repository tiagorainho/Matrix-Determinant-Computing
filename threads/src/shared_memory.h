#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <stdbool.h>

#include "fifo.h"

/**
 *  \file shared_memory.h
 * 
 *  Where the files syncronize and there is assured thread safety
 * 
 */

/** \brief Represents the FileHandler */
typedef struct sFileHandler {
    char * fileName;
    double * determinants;
    unsigned int nMatrices;
    unsigned int order;
} FileHandler;


/** \brief Initiates the structure
 * 
 *  Initiates the structure
 * 
 *  \param fileNames list of names of files
 *  \param fileCount files length
 * 
 */
void sm_init(char ** fileNames, unsigned int fileCount);


/** \brief Fetches a File from the FileHandlers
 * 
 *  Used by the file reader thread to fetch which file to read the matrices from
 * 
 *  \param[out] fileHandler Pointer to a FileHandler used in the worker thread
 *  \param[out] fileIdx Index of the current file
 *  
 *  \return boolean representing True if there is more work for the thread or False if the thread can exit.
 */
extern bool sm_getFile(FileHandler ** fileHandler, unsigned int * fileIdx);


/** \brief Adds a MatrixHandler to the Fifo
 * 
 *  Used by the file reader thread to add the matrices
 * 
 *  \param matrixHandler list of names of files
 *  
 */
extern void sm_addMatrix(MatrixHandler matrixHandler);


/** \brief Fetches a MatrixHandler from the Fifo
 * 
 *  Used by the determinant computing thread
 * 
 *  \param[out] matrixHandler Pointer to a FileHandler used in the worker thread
 *  
 */
extern bool sm_getMatrix(MatrixHandler * matrixHandler);


/** \brief Registers the result on the FileHandler
 * 
 *  Used by the determinant computing thread to registry its computation.
 * 
 *  \param matrixHandler Pointer to a FileHandler
 *  \param result Computed determinant
 *  
 */
extern void sm_registerResult(MatrixHandler * matrixHandler, double result);

/** \brief Get results from all the FileHandlers
 * 
 *  Provides all the FileHandlers it contains
 * 
 *  \param[out] results Array of FileHandlers
 *  
 */
extern void sm_getResults(FileHandler ** results);

/** \brief Frees the used memory    */
extern void sm_close();

#endif