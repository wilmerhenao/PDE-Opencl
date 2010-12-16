#include "cl-helper.h"
#include <math.h>




// #define DO_TIMING
// #define FOURTH_ORDER
#ifndef VERSION
#error Need to specify VERSION.
#endif

#define ORDER 2

#define GHOST_COUNT (ORDER/2)

#ifdef USE_DOUBLE
typedef double ftype;
#else
typedef float ftype;
#endif

#ifdef USE_ALIGNMENT
int use_alignment = 1;
#else
int use_alignment = 0;
#endif



double square(double x)
{
  return x*x;
}




int main()
{
#ifdef DO_TIMING
  double gbytes_accessed = 0;
  double seconds_taken = 0;
  double mcells_updated = 0;
  double gflops_performed = 0;
#endif

  // print_platforms_devices();

  cl_context ctx;
  cl_command_queue queue;
  create_context_on("NVIDIA", NULL, 0, &ctx, &queue,
#ifdef DO_TIMING
      1
#else
      0
#endif
      );

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  // read the cl file
  char buf[100];
  sprintf(buf, "wave-kernel-o%d-ver%d.cl", ORDER, VERSION);
  char *knl_text = read_file(buf);
  //get work group dimensions and gflop info.
  int wg_dims , wg_x, wg_y, wg_z, z_div, fetch_per_pt, flops_per_pt;
  if (sscanf(knl_text, "// workgroup: (%d,%d,%d) z_div:%d fetch_per_pt:%d flops_per_pt:%d", 
        &wg_x, &wg_y, &wg_z, &z_div, &fetch_per_pt, &flops_per_pt) == 6)
  {
    wg_dims = 3;
  }
  else if (sscanf(knl_text, "// workgroup: (%d,%d) fetch_per_pt:%d flops_per_pt:%d",
        &wg_x, &wg_y, &fetch_per_pt, &flops_per_pt) == 4)
  {
    wg_dims = 2;
    wg_z = -1;
    z_div = -1;
  }
  else
  {
    perror("reading workgroup spec");
    abort();
  }
#ifdef USE_DOUBLE
  char *compile_opt = "-DFTYPE=double";
#else
  char *compile_opt = "-DFTYPE=float";
#endif

  // creation of gthe kernel
  cl_kernel wave_knl = kernel_from_string(ctx, knl_text, "fd_update", 
      compile_opt);
  free(knl_text);
  // creation of the other kernel (the one that adds a number to a vector)
  knl_text = read_file("source-term.cl");
  cl_kernel source_knl = kernel_from_string(ctx, knl_text, "add_source_term", 
      compile_opt);
  free(knl_text);

  // --------------------------------------------------------------------------
  // set up grid
  // --------------------------------------------------------------------------
  const unsigned points = POINTS;

  const ftype minus_bdry = -1, plus_bdry = 1;

  // We're dividing into (points-1) intervals.
  ftype dx = (plus_bdry-minus_bdry)/(points-1);
  ftype dt = 0.5*dx;
  ftype dt2_over_dx2 = dt*dt / (dx*dx);

  // I can erase this part in my own project
  ftype final_time;
#ifdef DO_TIMING
  if (points > 256)
    final_time = 0.05;
  else if (points > 128)
    final_time= 0.05;
  else 
    final_time = 0.5;
#else
  const ftype final_time = 20;
#endif

  // This might run a little short, which is ok in our case.
  unsigned step_count = final_time/dt;
#ifndef DO_TIMING
  printf("will take %d steps.\n", step_count);
#endif

  // --------------------------------------------------------------------------
  // allocate and initialize CPU memory
  // --------------------------------------------------------------------------

  unsigned dim_other = points+2*GHOST_COUNT; //if order 2 then 1 point extra on each side
#ifdef USE_ALIGNMENT
  unsigned dim_x = ((dim_other + 15) / 16) * 16; // adjusts dimension to the next number divisible by 16
  unsigned field_start = 16 + GHOST_COUNT*(dim_x+dim_other*dim_x);
#else
  unsigned dim_x = dim_other;
  unsigned field_start = GHOST_COUNT*(1+dim_x+dim_other*dim_x);// this one puts me right at the beginning (the first positions get ignored cuz they are ghost points
#endif

  const size_t field_size = 16+dim_x*dim_other*dim_other;
  ftype *host_buf = malloc(field_size*sizeof(ftype));
  CHECK_SYS_ERROR(!host_buf, "allocating host_buf");

  // --------------------------------------------------------------------------
  // allocate GPU memory
  // --------------------------------------------------------------------------
  cl_int status;
  cl_mem dev_buf_a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
      sizeof(ftype) * field_size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem dev_buf_b = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      sizeof(ftype) * field_size, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // --------------------------------------------------------------------------
  // validate
  // --------------------------------------------------------------------------
  if (points < 128)
  {
    // zero out dev_buf_a
    for (size_t i = 0; i < field_size; ++i)
      host_buf[i] = 0;

    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
          queue, dev_buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
          field_size * sizeof(ftype), host_buf,
          0, NULL, NULL));

    // set up a field
    for (size_t i = 0; i < points; ++i)
      for (size_t j = 0; j < points; ++j)
        for (size_t k = 0; k < points; ++k)
        {
   	  // el cubo se llena primero de forma ascendiente y luego por el eje y.
	  // notese que los campos de la cola se "invaden" el principio de la siguiente fila
          unsigned base = field_start + i + dim_x*(j + dim_other * k);

          ftype x = i * dx;
          ftype y = j * dx;
          ftype z = k * dx;
          host_buf[base] = exp(x)*sin(y)*cos(z);
        }
    // variable f is assigned to buffer b
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
          queue, dev_buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
          field_size * sizeof(ftype), host_buf,
          0, NULL, NULL));

    // compute fd on device, read into host_buf_2
    ftype *host_buf_2 = malloc(field_size*sizeof(ftype));
    CHECK_SYS_ERROR(!host_buf, "allocating host_buf_2");

    {
      size_t gdim[] = { points, points, points/z_div };
      size_t ldim[] = { wg_x, wg_y, wg_z };

      SET_8_KERNEL_ARGS(wave_knl, dt2_over_dx2, dev_buf_a, dev_buf_b, dev_buf_a, 
          field_start, dim_x, dim_other, points);

      CALL_CL_GUARDED(clEnqueueNDRangeKernel,
          (queue, wave_knl,
           /*dimensions*/ wg_dims, NULL, gdim, ldim,
           0, NULL, NULL));
    }

    CALL_CL_GUARDED(clEnqueueReadBuffer, (
          queue, dev_buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
          field_size * sizeof(ftype), host_buf_2,
          0, NULL, NULL));

    // compute fd on host
    ftype *u = host_buf;

    double err_sum = 0;
    for (size_t i = 0; i < points; ++i)
      for (size_t j = 0; j < points; ++j)
        for (size_t k = 0; k < points; ++k)
        {
          unsigned base = field_start + i + dim_x*(j + dim_other * k);

          ftype ref_value =
            2 * u[base] - /*hist_u[base] */ 0
            + dt2_over_dx2 * (
                - 6*u[base]
                + u[base - 1]
                + u[base + 1]
                + u[base - dim_x]
                + u[base + dim_x]
                + u[base - dim_x*dim_other]
                + u[base + dim_x*dim_other]
                );

          err_sum += square(ref_value-host_buf_2[base]);// comparing the error with the cpu
        }
    free(host_buf_2);

    if (err_sum > 1e-8)
    {
      printf("excessive error: %g\n", err_sum);
      abort();
    }
  }
  // NOW THE CASE WHEN POINTS > 128... and the other too.
  // --------------------------------------------------------------------------
  // zero out arrays
  // --------------------------------------------------------------------------
  for (size_t i = 0; i < field_size; ++i)
    host_buf[i] = 0;

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, dev_buf_a, /*blocking*/ CL_TRUE, /*offset*/ 0,
        field_size * sizeof(ftype), host_buf,
        0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, dev_buf_b, /*blocking*/ CL_TRUE, /*offset*/ 0,
        field_size * sizeof(ftype), host_buf,
        0, NULL, NULL));

  // --------------------------------------------------------------------------
  // time step loop
  // --------------------------------------------------------------------------
  cl_mem cur_u = dev_buf_a, hist_u = dev_buf_b;

  for (unsigned step = 0; step < step_count; ++step)
  {
    ftype t = step * dt;
    if (step % 100 == 0)
    {
#ifndef DO_TIMING
      printf("step %d\n", step);
#endif
      CALL_CL_GUARDED(clFinish, (queue));
    }

    // ------------------------------------------------------------------------
    // visualize, if necessary
    // ------------------------------------------------------------------------
#ifndef DO_TIMING
    if (step % (points / 8) == 0 && t > 0.6)
    {
      CALL_CL_GUARDED(clEnqueueReadBuffer, (
            queue, cur_u, /*blocking*/ CL_TRUE, /*offset*/ 0,
            field_size * sizeof(ftype), host_buf,
            0, NULL, NULL));

      for (size_t i = 0; i < field_size; ++i)
        if (isnan(host_buf[i]))
        {
          fputs("nan encountered, aborting\n", stderr);
          abort();
        }

      char fnbuf[100];
      sprintf(fnbuf, "wave-%05d.bov", step);

      FILE *bov_header = fopen(fnbuf, "w");
      CHECK_SYS_ERROR(!bov_header, "opening vis header");

      sprintf(fnbuf, "wave-%05d.dat", step);
      fprintf(bov_header, "TIME: %g\n", t);
      fprintf(bov_header, "DATA_FILE: %s\n", fnbuf);
      fprintf(bov_header, "DATA_SIZE: %d %d %d\n", dim, dim, dim);
      fprintf(bov_header, "DATA_FORMAT: %s\n", sizeof(ftype) == sizeof(float) ? "FLOAT" : "DOUBLE");
      fputs("VARIABLE: solution\n", bov_header);
      fputs("DATA_ENDIAN: LITTLE\n", bov_header);
      fputs("CENTERING: nodal\n", bov_header);
      fprintf(bov_header, "BRICK_ORIGIN: %g %g %g\n", minus_bdry, minus_bdry, minus_bdry);
      fprintf(bov_header, "BRICK_SIZE: %g %g %g\n", 
          plus_bdry-minus_bdry,
          plus_bdry-minus_bdry,
          plus_bdry-minus_bdry);
      fclose(bov_header);

      FILE *bov_data = fopen(fnbuf, "wb");
      CHECK_SYS_ERROR(!bov_header, "opening vis output");
      fwrite((void *)host_buf, sizeof(ftype), field_size, bov_data);
      fclose(bov_data);
    }
#endif

    {
      // ----------------------------------------------------------------------
      // invoke wave kernel
      // ----------------------------------------------------------------------
      size_t gdim[] = { points, points, points/z_div };
      size_t ldim[] = { wg_x, wg_y, wg_z };

      SET_8_KERNEL_ARGS(wave_knl, dt2_over_dx2, hist_u, cur_u, hist_u, 
          field_start, dim_x, dim_other, points);

#ifdef DO_TIMING
      cl_event evt;
      cl_event *evt_ptr = &evt;
#else
      cl_event *evt_ptr = NULL;
#endif

      CALL_CL_GUARDED(clEnqueueNDRangeKernel,
          (queue, wave_knl,
           /*dimensions*/ wg_dims, NULL, gdim, ldim,
           0, NULL, evt_ptr));

#ifdef DO_TIMING
      // If timing is enabled, this wait can mean a significant performance hit.
      CALL_CL_GUARDED(clWaitForEvents, (1, &evt));

      cl_ulong start, end;
      CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, 
            CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
      CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, 
            CL_PROFILING_COMMAND_END, sizeof(start), &end, NULL));

      gbytes_accessed += 1e-9*(sizeof(ftype) * field_size * fetch_per_pt);
      seconds_taken += 1e-9*(end-start);
      mcells_updated += points*points*points/1e6;
      gflops_performed += 1e-9*points*points*points * flops_per_pt;

      CALL_CL_GUARDED(clReleaseEvent, (evt));
      CALL_CL_GUARDED(clFinish, (queue));
#endif
    }

    {
      // ----------------------------------------------------------------------
      // invoke source term kernel
      // ----------------------------------------------------------------------
      size_t gdim[] = { 1 };
      size_t ldim[] = { 1 };

      unsigned base = field_start + 
        (points/4) + dim_x*((points/5) + dim_other * (points/6));
      ftype value = dt*dt*sin(20*t);
      SET_3_KERNEL_ARGS(source_knl, hist_u, base, value);

      CALL_CL_GUARDED(clEnqueueNDRangeKernel,
          (queue, source_knl,
           /*dimensions*/ 1, NULL, gdim, ldim,
           0, NULL, NULL));
      CALL_CL_GUARDED(clFinish, (queue));
    }

    // swap buffers
    cl_mem tmp = cur_u;
    cur_u = hist_u;
    hist_u = tmp;
  }

#ifdef DO_TIMING
  printf("order:%d ftype:%d ver:%d align:%d pts:%d\tgflops:%.1f\tmcells:%.1f\tgbytes:%.1f [/sec]\n",
      ORDER, (int) sizeof(ftype), VERSION, use_alignment, points,
      gflops_performed/seconds_taken,
      mcells_updated/seconds_taken,
      gbytes_accessed/seconds_taken);
#endif

  CALL_CL_GUARDED(clFinish, (queue));

  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_GUARDED(clReleaseMemObject, (dev_buf_a));
  CALL_CL_GUARDED(clReleaseMemObject, (dev_buf_b));
  CALL_CL_GUARDED(clReleaseKernel, (wave_knl));
  CALL_CL_GUARDED(clReleaseKernel, (source_knl));
  CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
  CALL_CL_GUARDED(clReleaseContext, (ctx));

  free(host_buf);
}
