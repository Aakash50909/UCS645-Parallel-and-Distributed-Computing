all: problem1 problem2 problem3

problem1:
	nvcc -O2 problem1.cu -o problem1

problem2:
	nvcc -O2 problem2.cu -o problem2

problem3:
	nvcc -O2 problem3.cu -o problem3

clean:
	rm -f problem1 problem2 problem3
