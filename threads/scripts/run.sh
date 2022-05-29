
dataset_src='../dataset'
../bin/determinant_calculation.out \
    -f $dataset_src/mat128_32.bin \
    :'
    -f $dataset_src/mat128_64.bin \
    -f $dataset_src/mat128_128.bin \
    -f $dataset_src/mat128_256.bin \
    -f $dataset_src/mat512_32.bin \
    -f $dataset_src/mat512_64.bin \
    -f $dataset_src/mat512_128.bin \
    -f $dataset_src/mat512_256.bin \
    '