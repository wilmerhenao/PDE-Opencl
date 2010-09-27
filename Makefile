#
#  Makefile for openMP assignment
#
CMD    = xserial
CC     = icc
CFLAGS = -O2 -openmp 
LFLAGS = -openmp 
LIBS   = -lm -lrt
OBJS   = serial_hw2.o timing.o 

.c.o:
	$(CC)  $(CFLAGS) $(INCLUDE) -c $<


$(CMD): $(OBJS)
	$(CC)  -o $@ $^ $(LFLAGS) $(LIBS)

# 
.PHONY: clean new

clean:
	-/bin/rm -f *.o *~ $(CMD)

new:
	make clean
	make $(CMD)
