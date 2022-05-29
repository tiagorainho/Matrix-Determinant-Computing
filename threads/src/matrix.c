
#include <stdio.h>

#include "matrix.h"


void print_matrix(Matrix * matrix) {
    for(int i=0; i<matrix->order; i++) {
        printf("%d\n", i);
        for(int j=0; j<matrix->order; j++) {
            printf("%f  ", matrix->numbers[i][j]);
        }
        printf("\n");
    }
}

void switch_row(Matrix matrix, int row1, int row2) {
    double aux;
    for(int i=0;i<matrix.order;i++) {
        aux = matrix.numbers[row1][i];
        matrix.numbers[row1][i] = matrix.numbers[row2][i];
        matrix.numbers[row2][i] = aux;
    }
}

double compute_determinant(Matrix matrix) {
    int sign = 1;
    double ratio, determinant = 1;
    
    for(int i=0;i<matrix.order;i++) {
        // check if the row can be used, otherwise, switch that row
        if(matrix.numbers[i][i] == 0) {
            for(int j=i+1;j<matrix.order;j++) {
                if(matrix.numbers[j][i] != 0) {
                    switch_row(matrix, i, j);
                    sign = (sign == 1) ? -1: 1;
                    break;
                }
            }                
        }

        for(int j=i+1;j<matrix.order;j++) {
            ratio = matrix.numbers[j][i]/matrix.numbers[i][i];
            for(int k=0;k<matrix.order;k++) {
                matrix.numbers[j][k] = matrix.numbers[j][k]-ratio*matrix.numbers[i][k];
            }
        }
        determinant *= matrix.numbers[i][i];
    }

    return determinant * sign;
}