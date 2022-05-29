
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


/** \brief Constructs a matrix
 * 
 *  \param order order of the matrix
 *  \param numbers pointer to pointers of rows of a matrix
 *  \param[out] matrix the pointer of the matrix constructed
 * 
*/
Matrix * init_matrix(int order, double numbers[order][order]);


/** \brief Clean matrix memory
 *  
 *  \param matrix the pointer of the matrix to clean
 * 
*/
void free_matrix(Matrix * matrix);

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