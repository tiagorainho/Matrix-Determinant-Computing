
src='../src'
gcc $src/main.c $src/determinant_calculation.c $src/file_reader.c $src/matrix.c $src/shared_memory.c $src/fifo.c -o ../bin/determinant_calculation.out -lpthread