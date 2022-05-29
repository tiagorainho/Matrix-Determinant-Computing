
#ifndef FIFO_H
#define FIFO_H

#include <stdio.h>

#include "matrix.h"


typedef struct sMatrixHandler {
    Matrix * matrix;
    int fileIdx;
    int matrixIdx;
} MatrixHandler;


/** \brief Inserts a matrix inside the fifo
 *  
 *  \param id Worker thread id.
 *  \param[in] MatrixHandler Matrix containing both the order and quocients of a matrix.
 */
extern void putMatrix(unsigned int producerId, MatrixHandler * matrix);


/** \brief Fetches a matrix from the fifo
 *  
 *  \param bool boolean which if TRUE, represents that there is more work for the thread, if FALSE, means that the thread can exit.
 *  \param[in] receiverId Worker thread id.
 *  \param[out] matrix Matrix pointer.
 */
extern bool getMatrix(unsigned int receiverId, MatrixHandler ** matrix);

/** \brief Signal for the Fifo
 *  
 *  This signal represents that there is no more matrices available to insert into the memory.
 */
extern void doneReading();

#endif /* FIFO_H */
