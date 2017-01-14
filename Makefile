CC = nvcc -rdc=true -arch=sm_20 -ccbin mpicxx
SRCS = $(wildcard *.cpp)
CUSRCS = $(wildcard *.cu)
HDRS = $(wildcard *.h)
PROJ = main

APP = $(PROJ)

all: $(APP)

$(APP): $(HDRS) $(SRCS) $(CUSRCS)
	$(CC) $(SRCS) $(CUSRCS) -o $(APP)

clean:
	rm -f *.o $(APP)
