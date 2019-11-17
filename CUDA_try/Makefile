TARGET=nn
SRCS = main.cpp

OBJS = $(SRCS:.c=.o)

$(TARGET):
	nvcc -std=c++14 -o main main.cu

clean:
	rm main
