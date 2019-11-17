TARGET=nn
SRCS = library.cpp

OBJS = $(SRCS:.c=.o)

$(TARGET):
	nvcc -std=c++11 -Xcompiler -fPIC -O2 -c matmul.cu
	g++ -std=c++11 -shared -O3 -DNDEBUG -fPIC -o nn_c.so  matmul.o library.cpp -Wl,-Bdynamic -lpthread -Wl,-Bstatic -lopenblas -Wl,-Bdynamic -lcudart -Wl,--as-needed -m64
	mv -f ./nn_c.so ../
	# nvcc -std=c++11 -O2 -c matmul.cu
	# g++ -std=c++11 -o main matmul.o library.cpp -Wl,-Bstatic -lopenblas -Wl,-Bdynamic -lpthread  -Wl,-Bdynamic -lcudart -Wl,--as-needed -m64
#	g++ -std=c++11 -shared -O3 -DNDEBUG -fPIC -o nn_c.so matmul.o library.cpp -Wl,-Bdynamic -lpthread -Wl,-Bstatic -lopenblas -Wl,-Bdynamic -lcudart -Wl,--as-needed -m64
#	g++ -std=c++14 -shared -O2 -fPIC -o nn_c.so library.cpp libopenblas.a -lpthread
