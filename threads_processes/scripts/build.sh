
src='../src'
mpicc $src/main.c $src/matrix.c $src/shared_memory.c $src/fifo.c -o ../bin/determinant_calculation.out -lpthread