all: ex01 ex02 ex03 ex04

ex01:
	nvcc -O2 -arch=sm_86 ex01/ex01_cuda_basics.cu -o ex01/ex01 -lm

ex02:
	nvcc -O2 -arch=sm_86 ex02/ex02_memory_hierarchy.cu -o ex02/ex02 -lm

ex03:
	nvcc -O2 -arch=sm_86 ex03/ex03_ml_primitives.cu -o ex03/ex03 -lm

ex04:
	nvcc -O2 -arch=sm_86 ex04/ex04_cnn_layers.cu -o ex04/ex04 -lcublas

clean:
	rm -f ex01/ex01 ex02/ex02 ex03/ex03 ex04/ex04
