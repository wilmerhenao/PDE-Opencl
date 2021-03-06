# /----------------------------------------------------------
# | General setup
# +----------------------------------------------------------
# |
CC = cc
CFLAGS = -std=c99 -Wall -Werror -D_XOPEN_SOURCE=500 #-v -Q
OPTFLAGS = -g
LDFLAGS = -lOpenCL -lm #--enable-checking
DEFINES = -DVERSION=0 -DPOINTS=64
# |
# \----------------------------------------------------------

# /----------------------------------------------------------
# | Compilation rules
# +----------------------------------------------------------
# |
#
EXECUTABLES = hello-gpu-small hello-gpu mg mg2

.PHONY:	all
all:	$(EXECUTABLES)

# Compile a C version (using basic_dgemm.c, in this case):

hello-gpu: hello-gpu.o cl-helper.o
	$(CC) -o $@ $^ 

mg: mg.o cl-helper.o
	$(CC) -o $@ $^ $(LDFLAGS)

mg2: mg2.o cl-helper.o
	$(CC) -o $@ $^ $(LDFLAGS)

# Generic Rules
%.o:%.c
	$(CC) -c $(CFLAGS) $(OPTFLAGS) $(DEFINES) $<
# |
# \----------------------------------------------------------

# /----------------------------------------------------------
# | Clean-up rules
# +----------------------------------------------------------

.PHONY:	clean
clean:
	rm -f $(EXECUTABLES) *.o
# |
# \----------------------------------------------------------

