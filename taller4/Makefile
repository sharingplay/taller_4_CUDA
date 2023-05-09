CC = g++
CFLAGS = -std=c++11
INCLUDES = -I/usr/local/include/opencv4 -I/usr/local/cuda/include
LIBS = -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lcudart

all: image_processing

image_processing: calc_pi.cu
    $(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

clean:
    rm -f image_processing
