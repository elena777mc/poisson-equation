CC = mpicxx
SRCS = $(wildcard *.cpp)
HDRS = $(wildcard *.h)
OBJS = $(SRCS:.cpp=.o)
DIRS = $(subst /, ,$(CURDIR))
PROJ = main

APP = $(PROJ)
CFLAGS= -c -O3
LDFLAGS=
LIBS= -O3 -fopenmp

all: $(APP)

$(APP): $(OBJS)
	$(CC) $(INCLUDE_DIR)  $(LDFLAGS) $(OBJS) -o $(APP) $(LIBS)

%.o: %.cpp $(HDRS) $(MF)
	$(CC) $(INCLUDE_DIR)  $(CFLAGS) $< -o $@

clean:
	rm -f *.o $(APP)
