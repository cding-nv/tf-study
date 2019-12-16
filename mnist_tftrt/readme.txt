g++ -std=c++11 -I/home/cding/tensorflow/ -c ./test.cc -o test.o
g++ -o test ./test.o -ltensorflow_cc -ltensorflow_framework
./test ./mnist_frozen_graph.pb ArgMax,Softmax
