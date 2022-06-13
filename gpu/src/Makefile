APPS=matrixDeterminantRows matrixDeterminantCols
buildFolder=bin

all: ${APPS} 
%: %.cu
	mkdir -p ${buildFolder}
	nvcc -O2 -Wno-deprecated-gpu-targets -o ${buildFolder}/$@ $<

clean:
	rm -f -r ${buildFolder}
