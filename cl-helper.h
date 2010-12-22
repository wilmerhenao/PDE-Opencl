/*
 * Copyright (c) 2010 Andreas Kloeckner
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */




#ifndef NYUHPC_CL_HELPER
#define NYUHPC_CL_HELPER

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CHECK_CL_ERROR(STATUS_CODE, WHAT) \
  if ((STATUS_CODE) != CL_SUCCESS) \
  { \
    fprintf(stderr, \
        "*** '%s' in '%s' on line %d failed with error '%s'.\n", \
        WHAT, __FILE__, __LINE__, \
        cl_error_to_str(STATUS_CODE)); \
    abort(); \
  }

#define CALL_CL_GUARDED(NAME, ARGLIST) \
  { \
    cl_int status_code; \
      status_code = NAME ARGLIST; \
    CHECK_CL_ERROR(status_code, #NAME); \
  }

#define CHECK_SYS_ERROR(COND, MSG) \
  if (COND) \
  { \
    perror(MSG); \
    abort(); \
  }

const char *cl_error_to_str(cl_int e);
void print_platforms_devices();
void create_context_on(
    const char *plat_name, const char*dev_name, cl_uint idx,
    cl_context *ctx, cl_command_queue *queue, int enable_profiling);
char *read_file(const char *filename);
cl_kernel kernel_from_string(cl_context ctx, 
    char const *knl, char const *knl_name, char const *options);

#define SET_1_KERNEL_ARG(knl, arg0) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0));

#define SET_2_KERNEL_ARGS(knl, arg0, arg1) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1));

#define SET_3_KERNEL_ARGS(knl, arg0, arg1, arg2) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(arg2), &arg2));

#define SET_4_KERNEL_ARGS(knl, arg0, arg1, arg2, arg3) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(arg2), &arg2)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 3, sizeof(arg3), &arg3));

#define SET_5_KERNEL_ARGS(knl, arg0, arg1, arg2, arg3, arg4) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(arg2), &arg2)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 3, sizeof(arg3), &arg3)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 4, sizeof(arg4), &arg4));

#define SET_6_KERNEL_ARGS(knl, arg0, arg1, arg2, arg3, arg4, arg5) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(arg2), &arg2)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 3, sizeof(arg3), &arg3)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 4, sizeof(arg4), &arg4)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 5, sizeof(arg5), &arg5));

#define SET_7_KERNEL_ARGS(knl, arg0, arg1, arg2, arg3, arg4, arg5, arg6) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(arg2), &arg2)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 3, sizeof(arg3), &arg3)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 4, sizeof(arg4), &arg4)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 5, sizeof(arg5), &arg5)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 6, sizeof(arg6), &arg6));

#define SET_8_KERNEL_ARGS(knl, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(arg2), &arg2)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 3, sizeof(arg3), &arg3)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 4, sizeof(arg4), &arg4)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 5, sizeof(arg5), &arg5)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 6, sizeof(arg6), &arg6)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 7, sizeof(arg7), &arg7));

#define SET_9_KERNEL_ARGS(knl, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg9) \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 0, sizeof(arg0), &arg0)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 1, sizeof(arg1), &arg1)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 2, sizeof(arg2), &arg2)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 3, sizeof(arg3), &arg3)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 4, sizeof(arg4), &arg4)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 5, sizeof(arg5), &arg5)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 6, sizeof(arg6), &arg6)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 7, sizeof(arg7), &arg7)); \
  CALL_CL_GUARDED(clSetKernelArg, (knl, 8, sizeof(arg8), &arg8));




#endif
