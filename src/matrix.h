
#ifndef MATRIX_H
#define MATRIX_H

/**
 *  \file matrix.h
 * 
 */

/** \brief Represents the Matrix */
typedef struct sMatrix {
    unsigned int order;
    double ** numbers;
} Matrix;


/** \brief Prints the matrix
 *  
 *  \param matrix pointer of the matrix to be printed
 * 
*/
void print_matrix(Matrix * matrix);


/** \brief Computes the determinant of a matrix
 *  
 *  \param matrix Matrix to be used
 * 
 *  \returns double value of the determinant of the matrix given as input
*/
double compute_determinant(Matrix matrix);



#endif /* MATRIX_H */