g++ -std=c++11 -c ./tf.cpp  -o tf.o
g++ -o tf ./tf.o -ltensorflow_cc -ltensorflow_framework 
./tf ./digit.jpg 
