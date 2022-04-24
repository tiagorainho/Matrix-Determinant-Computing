

#ifndef DETERMINANT_WORKER
#define DETERMINANT_WORKER

/**
 *  \file determinant_calculation.h
 *
 *  \brief Determinant Computing Thread Worker implementation
 *
 *  This thread fetches matrices and proceeds to compute the matrices inside each file returning the result.
 *  
 *  The thread continues its work until there is no more matrices available
 * 
 *  \author João Diogo Ferreira, João Tiago Rainho - April 2022
 */


/** \brief worker which computes matrices
 *  
 *  Operation carried out by worker thread.
 *  Reads the matrices from the fifo.
 *
 *  \param arg Worker thread id
 */
void * compute_determinant_thread_worker(void * arg);

#endif