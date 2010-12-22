#include "cl-helper.h"
#include <math.h>

// #define DO_TIMING
// #define FOURTH_ORDER
#ifndef VERSION
#error Need to specify VERSION.
#endif

#define ORDER 1

#ifndef CUTOFF
#define CUTOFF 0
#endif

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

/* ----- local macros  -------------------------------------------------- */

#define INDEX3D(i,j,k)    field_start + i + dim_x*(j + dim_other * k)
#define MAX(A,B)          ((A) > (B)) ? (A) : (B)
#define ABS(A)            ((A) >= 0) ? (A) : -(A)
#define F(i,j,k)          f[INDEX3D(i,j,k)]
#define U(i,j,k)          u[INDEX3D(i,j,k)]
#define R(i,j,k)      	  r[INDEX3D(i,j,k)]
#define UEXACT(i,j,k)     uexact[INDEX3D(i,j,k)]
#define U_OLD(i,j,k)      u_old[INDEX3D(i,j,k)]
//printf("im here");
/* -----               -------------------------------------------------- */
/* ---- linked list for the different grids ----------------------------- */
// Each struct member contains a vector as well as values associated with it
// the size of tvec is maximum in order to fit everything.  Although a dynamically
// allocated vector would have worked a lot neater.

struct grids {
   ftype * uvec;
   ftype * fvec;
   ftype * rvec;
   unsigned field_start;
   unsigned dim_x;
   unsigned dim_other;
   struct grids * next;
   struct grids * prev;
};

typedef struct grids item;

/* ----- function declarations ------------------------------------------ */
void init_f (unsigned, ftype [], ftype, unsigned, unsigned, unsigned, ftype);
void init_u ( unsigned, ftype [], ftype,  ftype , ftype, unsigned, unsigned, unsigned );
void init_uexact(unsigned, ftype [], ftype [], ftype, size_t, unsigned, unsigned, unsigned);
ftype ffun (ftype, ftype, ftype);
ftype ufun (ftype, ftype, ftype);
void resid ( ftype [], ftype [], ftype [], ftype, size_t, unsigned, unsigned, unsigned);
void resid2(struct grids *, ftype );
void ctof(struct grids * , struct grids * , size_t);
void gsrelax(struct grids * , ftype );
void U_error(ftype [], ftype [], ftype [], unsigned);
ftype norm(ftype [], unsigned );
void injf2c(struct grids * , struct grids *);
void mgv(ftype [], ftype [], ftype, unsigned, unsigned, unsigned, size_t, unsigned, int, unsigned, cl_context, cl_command_queue, cl_kernel, int, int, int , int, int, int, int);
/* ---- -------------------------------------------------------------------*/
#ifdef DO_TIMING
  double gbytes_accessed = 0;
  double seconds_taken = 0;
  double mcells_updated = 0;
  double gflops_performed = 0;
  double tot_secs = 0;
#endif

int main()
{
  int enable_profiling = 0;
  #ifdef DO_TIMING
      enable_profiling = 1;
  #endif

  //print_platforms_devices();
  cl_context ctx;
  cl_command_queue queue;
  create_context_on("NVIDIA", NULL, 0, &ctx, &queue, enable_profiling);

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  // read the cl file

  char buf[100];
  sprintf(buf, "mg-kernel-ver%d.cl", VERSION);
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

  // creation of the kernel
  cl_kernel poisson_knl = kernel_from_string(ctx, knl_text, "fd_update", compile_opt);
  free(knl_text); // my compiler complains about this one.  OJO!!
  // --------------------------------------------------------------------------
  // set up grid
  // --------------------------------------------------------------------------
  const unsigned points = POINTS;
  const ftype minus_bdry = -1, plus_bdry = 1;

  // We're dividing into (points-1) intervals.
  ftype dx = (plus_bdry-minus_bdry)/(points-1);

  // --------------------------------------------------------------------------
  // allocate and initialize CPU memory
  // --------------------------------------------------------------------------
  int use_alignment;
  unsigned dim_other = points; //if order 2 then 1 point extra on each side
  #ifdef USE_ALIGNMENT
  // adjusts dimension so that the next row starts in a number divisible by 16
  unsigned dim_x = ((dim_other + 15) / 16) * 16; 
  unsigned field_start = 0;
  use_alignment = 1; 
  #else
  unsigned dim_x = dim_other;
  unsigned field_start = 0;// this one puts me right at the beginning
  use_alignment = 0;
  #endif
  // --------Allocate forcing uexact, r and u vectors -------------------------
  const size_t field_size = 0+dim_x*dim_x*dim_x;  // extra large to fit the 2^n constrain in GPU
  ftype *f = malloc(field_size*sizeof(ftype));
  CHECK_SYS_ERROR(!f, "allocating f");
  ftype *u = malloc (field_size*sizeof(ftype));
  CHECK_SYS_ERROR(!u, "allocating u");  
  ftype *uexact = malloc (field_size*sizeof(ftype));
  CHECK_SYS_ERROR(!uexact, "allocating uexact");
  ftype *r = malloc(field_size * sizeof(ftype));
  CHECK_SYS_ERROR(!r, "allocating residual r");

  // --------------------------------------------------------------------------
  // initialize
  // --------------------------------------------------------------------------
    // zero out (necessary to initialize everything bec. I measure norms)
    for (size_t i = 0; i < field_size; ++i){
      f[i] = 0;
      u[i] = 0;
      uexact[i] = 0;
      r[i] = 0;
    }
    // set up the forcing field
    init_f (points, f, dx, field_start, dim_x, dim_other, minus_bdry);
    // Initialize u with initial boundary conditions
    init_u ( points, u , minus_bdry, plus_bdry, dx, field_start, dim_x, dim_other);
    // Initialize the exact solution
    init_uexact(points, u, uexact, dx, field_size, field_start, dim_x, dim_other);

    // --------------------------------------------------------------------------
    // Setup the v-cycles
    // --------------------------------------------------------------------------
  
    unsigned n1, n2, n3, ncycles;
    n1 = 50;
    n2 = 60;
    n3 = 1;
    ncycles = 2;
    ftype *sweeps = malloc (ncycles*sizeof(ftype));
    ftype *rnorm = malloc (ncycles*sizeof(ftype));
    ftype *enorm = malloc (ncycles*sizeof(ftype));
    ftype rtol = 1.0e-05;

    // Find the norm of the residual (choose your method)
    sweeps[0] =0;
    resid (r, f, u, dx, field_size, field_start, dim_x, dim_other);
    rnorm[0] = norm( r , field_size) * dx;
    U_error(u, uexact, r, field_size);
    enorm[0] = norm( r, field_size ) * dx;

    for(unsigned icycle = 1; icycle <= ncycles; icycle++){
       mgv(f, u, dx, n1, n2, n3, field_size, points, use_alignment, dim_x, ctx, queue, poisson_knl, wg_dims , wg_x, wg_y, wg_z, z_div, fetch_per_pt, flops_per_pt);  //update u through a v-cycle 
       sweeps[icycle] = sweeps[icycle -1] + (4 * (n1 + n2)/3);
       resid (r, f, u, dx, field_size, field_start, dim_x, dim_other);
       rnorm[icycle] = norm( r, field_size ) * dx;
       U_error(u, uexact, r, field_size);
       enorm[icycle] = norm( r, field_size ) * dx;
       //cfacts = (rnorm(icycle)/rnorm(icycle - 1))^(1 / (n1 + n2)) not necessary
       //disp something here if I want to.
       //printf("norm of the cycle %f", enorm[icycle]);
       if(rnorm[icycle] <= rtol * rnorm[0])
	  break;
    }
    #ifdef DO_TIMING
  printf(" ftype:%d ver:%d align:%d pts:%d\tgflops:%.1f\tmcells:%.1f\tgbytes:%.1f [/sec]\tout_gflops:%.6f\n", (int) sizeof(ftype), VERSION, use_alignment, points, gflops_performed/seconds_taken, mcells_updated/seconds_taken, gbytes_accessed/seconds_taken, gflops_performed/tot_secs);
#endif
  // --------------------------------------------------------------------------
  // clean up
  // --------------------------------------------------------------------------
  CALL_CL_GUARDED(clReleaseKernel, (poisson_knl));
  CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
  CALL_CL_GUARDED(clReleaseContext, (ctx));
}

ftype ufun ( ftype x, ftype y, ftype z ){
  return (exp( x ) * exp( -2 * y ) * exp( z ));  // exact solution
  //return (cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return (cos(pi*x) + cos(pi*y) + cos(pi*z));
  //return 2.;
}

ftype ffun (ftype x, ftype y, ftype z){
  return ( 6 * exp(x) * exp(-2 * y) * exp(z));
}

/******************************************************************************/

void init_u ( unsigned points, ftype u[], ftype lo,  ftype hi, ftype dx , unsigned field_start, unsigned dim_x, unsigned dim_other)
{
  size_t i, j, k;
  ftype x, y, z;

    /* Set the boundary conditions. For this simple test use exact solution in bcs */

    j = 0;   // low y bc
    y = lo;

    for ( k = 0; k < points; k++ ) {
      z = lo + k*dx;
      for ( i = 0; i < points; i++ ) {
        x = lo + i * dx;
        U(i,j,k) = ufun (x,y,z);  // notese que cuando aca se habla de ijk, en realidad es una macro que asigna la posicion necesaria
      }
    }

    j = points - 1;  // hi y
    y = hi;
    
    for ( k = 0; k < points; k++ ) {
      z = lo + k*dx;
      for ( i = 0; i < points; i++ ) {
        x = lo + i * dx;
        U(i,j,k) = ufun ( x, y, z );
      }
    }

    i = 0;  // low x
    x = lo;

    for ( k = 0; k < points; k++ ) {
      z = lo + k*dx;
      for ( j = 0; j < points; j++ ) {
        y = lo + j * dx;
        U(i,j,k) = ufun ( x, y, z );
      }
    }

    i = points - 1; // hi x
    x = hi;

    for ( k = 0; k < points; k++ ) {
      z = lo + k*dx;    
      for ( j = 0; j < points; j++ ) {
        y = lo + j * dx;
        U(i,j,k) = ufun ( x, y, z );
      }
    }

    k = 0; // low z
    z = lo;

    for ( j = 0; j < points; j++ ) {
      y = lo + j * dx;
      for ( i = 0; i < points; i++) {
        x = lo + i * dx;
        U(i,j,k) = ufun ( x, y, z );
      }
    }

    k = points - 1; // hi z
    z = hi;
    
    for ( j = 0; j < points; j++ ) {
      y = lo + j * dx;
      for ( i = 0; i < points; i++) {
        x = lo + i * dx;
        U(i,j,k) = ufun ( x, y, z );
      }
    }

    return;
}

/* ------------------------------------------------------------------------- */
void init_f (unsigned points, ftype f[],ftype dx, unsigned field_start, unsigned dim_x, unsigned dim_other, ftype lo){

ftype x, y, z;
    
for (size_t i = 0; i < points; ++i)
   for (size_t j = 0; j < points; ++j)
     for (size_t k = 0; k < points; ++k){
	  // el cubo se llena primero de forma ascendiente y luego por el eje y.
	  // notese que los campos de la cola se "invaden" el principio de la siguiente fila
          //unsigned base = field_start + i + dim_x*(j + dim_other * k);
          x = lo + i * dx;
          y = lo + j * dx;
          z = lo + k * dx;
          F(i,j,k) = ffun(x, y, z);
      }
}

void resid ( ftype r[], ftype f[], ftype u[], ftype dx, size_t field_size, unsigned field_start, unsigned dim_x, unsigned dim_other){
  // This function computes the residual of the poisson problem
  // Input:  f is the right hand side
  // u current approximate solution
  // dx is the mesh spacing
  // Output: r is the residual
  
  size_t i, j, k;
  ftype h = dx * dx;
  for(i = 0; i < field_size; i++)
     r[i] = 0;

  for(i = 1; i < (POINTS-1); i++)
     for(j = 1; j < (POINTS-1); j++)
        for(k = 1; k < (POINTS-1); k++)
           R(i,j,k) = F(i,j,k) +  ( U(i+1,j,k) + U(i-1,j,k) + U(i,j-1,k) + U(i, j+1,k) + U(i,j,k-1) + U(i,j,k+1) - 6 * U(i,j,k)) / h;  //I have doubts with regard to this formula
}

void U_error(ftype u[], ftype uexact[], ftype thisdiff[], unsigned field_size){
  // find the distance to the exact solution
  for(size_t i = 0; i < field_size; i++)
	thisdiff[i] = u[i] - uexact[i];
}

void init_uexact(unsigned points, ftype u[], ftype uexact[], ftype dx, size_t field_size, unsigned field_start, unsigned dim_x, unsigned dim_other){
// this function calculates the exact value of the solution

  ftype lo = -1;
  ftype x, y, z;
  for(size_t i = 0; i < points; i++)
     for(size_t j = 0; j < points; j++)
        for(size_t k = 0; k < points; k++){
	  x = lo + i * dx;
          y = lo + j * dx;
          z = lo + k * dx;
          UEXACT(i,j,k) = ufun(x, y, z);
	}
}

void mgv(ftype f[], ftype u[], ftype dx, unsigned n1,unsigned n2,unsigned n3, size_t field_size, unsigned points, int use_alignment, unsigned dim_x, cl_context ctx, cl_command_queue queue, cl_kernel poisson_knl, int wg_dims , int wg_x, int wg_y, int wg_z, int z_div, int fetch_per_pt, int flops_per_pt){
  // mgv does one v-cycle for the Poisson problem on a grid with mesh size dx
  // Inputs: f is right hand side, u is current approx dx is mesh size, n1 number of sweeps
  // on downward branch, n2 number of sweeps on upwardbranch, n3 number of sweeps on
  // coarsest grid.
  // Output:  It just returns an updated version of u
  cl_ulong start_big;
  #ifdef DO_TIMING
  cl_event evt;
  cl_event *evt_ptr = &evt;
  #else
  cl_event *evt_ptr = NULL;
  #endif
  size_t i, isweep;
  item * ugrid, * head, * curr;
  int l = 0;
  ftype dxval[POINTS/2] = {0};  // this is huge and unnecessary.  Try to cut downif time!!  
  ftype h;
  unsigned nx[POINTS/2] = {0};
  // --- Allocate common gpu memory----
  cl_int status;
  cl_mem dev_buf_u = clCreateBuffer(ctx, CL_MEM_READ_WRITE, field_size * sizeof(ftype), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  cl_mem dev_buf_f = clCreateBuffer(ctx, CL_MEM_READ_ONLY, field_size * sizeof(ftype), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  cl_mem dev_buf_hist_u = clCreateBuffer(ctx, CL_MEM_READ_ONLY, field_size * sizeof(ftype), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  //cl_mem read_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, field_size * sizeof(ftype), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  // -----------------------------------
  dxval[0] = dx;
  nx[0] = points;
  //const size_t max_size  = POINTS * POINTS * ((POINTS + 15)/16) * 16;
  // --------------- Allocatig the finest grid --------------------
  ugrid = (item *)malloc(sizeof(item));
  ugrid->uvec = malloc(field_size * sizeof(ftype));
  ugrid->fvec = malloc(field_size * sizeof(ftype));
  ugrid->rvec = malloc(field_size * sizeof(ftype));
  ugrid->dim_other = nx[0];
  ugrid->dim_x = dim_x;
  for(i = 0; i < field_size; i++){
     ugrid->uvec[i] = u[i];
     ugrid->fvec[i] = f[i];
     ugrid->rvec[i] = 0;
  }
  head = ugrid;  // head will always be the first one

  // ---------------- Set up the coarse grids ----------------------
  while((nx[l] - 1) % 2 == 0 && nx[l] > 3){
    l = l+1;
    nx[l] = (nx[l - 1] - 1) / 2 + 1;
    dxval[l] = 2 * dxval[l-1]; 
    curr = (item *)malloc(sizeof(item));
    curr->uvec = malloc(field_size * sizeof(ftype));
    curr->fvec = malloc(field_size * sizeof(ftype));
    curr->rvec = malloc(field_size * sizeof(ftype));

    curr->dim_other = nx[l];
    curr->field_start = 0;
    curr->dim_x = curr->dim_other;
    if(use_alignment)
    	curr->dim_x = ((nx[l] + 15)/16) * 16;

    // initialize vectors
    for(i = 0; i < field_size; i++){
 	curr->uvec[i] = 0;
	curr->fvec[i] = 0;
     	curr->rvec[i] = 0;
    }
    ugrid->next = curr; // curr gets attached to ugrid
    curr->prev = ugrid;
    ugrid = curr;
  }
  int nl = l; // this is the maximum number of grids that were created
  // --- at this point head contains the finest grid and ugrid contains the coarsest -----
  curr = head;
  head->prev = NULL;
  ugrid->next = NULL;

  // ---------------- Now relax each of the different grids descending--------
  for(l = 0; l < nl; l++){  // I stop right before nl (will be treated different)
     // ----------------------------------------------------------------------
     // -------------------- GPU DESCENDING V-CYCLE --------------------------
     // ----------------------------------------------------------------------
     {
     if(curr->dim_other < CUTOFF){
	for(isweep = 0; isweep < n1; isweep++){
	     gsrelax(curr, dxval[l]);
	}
     }

     else{
  	// ---GPU------GPU------GPU------GPU------GPU------GPU------GPU------GPU--- //
  	// fill in the buffers inside the GPU with the current data
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_u, CL_TRUE, 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_f, CL_TRUE, 0, field_size * sizeof(ftype), curr->fvec, 0, NULL, NULL));
	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_hist_u, CL_TRUE, 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
  	h = dxval[l] * dxval[l];
        size_t gdim[] = { curr->dim_x, curr->dim_x, curr->dim_x/z_div };
        size_t ldim[] = { wg_x, wg_y, wg_z };

  	for(i = 0; i < n1; i++){
     	   // ----------------------------------------------------------------------
     	   // invoke poisson kernel
     	   // ----------------------------------------------------------------------
	   //size_t u_size;
	   //CALL_CL_GUARDED(clGetMemObjectInfo, (dev_buf_u, CL_MEM_SIZE, sizeof(u_size), &u_size, 0));
	   //int u_size_i = u_size;
	   //printf("u_size=%d fstart=%d dim_x=%d dim_other=%d\n" , u_size_i, curr->field_start, curr->dim_x, curr->dim_other);
	   curr->field_start = 0;
     	   SET_7_KERNEL_ARGS(poisson_knl, dev_buf_u, dev_buf_f, dev_buf_hist_u, curr->field_start, curr->dim_x, curr->dim_other, h);
     	   // run the kernel
     	   CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, poisson_knl, /*dimensions*/ wg_dims, NULL, gdim, ldim, 0, NULL, evt_ptr));
     	   #ifdef DO_TIMING
      	   // If timing is enabled, this wait can mean a significant performance hit.
      	   CALL_CL_GUARDED(clWaitForEvents, (1, &evt));
 
      	   cl_ulong start, end;
      	   CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
      	   CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, CL_PROFILING_COMMAND_END, sizeof(start), &end, NULL));
 
      	   gbytes_accessed += 1e-9*(sizeof(ftype) * field_size * fetch_per_pt);
	   start_big = start;
      	   seconds_taken += 1e-9*(end-start);
      	   mcells_updated += curr->dim_other*curr->dim_other*curr->dim_other/1e6;
      	   gflops_performed += 1e-9*curr->dim_x*curr->dim_x*curr->dim_x * flops_per_pt;

      	   CALL_CL_GUARDED(clReleaseEvent, (evt));
     	   #endif
      	   CALL_CL_GUARDED(clFinish, (queue)); //ira adentro??
	   cl_mem tmp = dev_buf_u;
	   dev_buf_u = dev_buf_hist_u;
	   dev_buf_hist_u = tmp;
     	}
        //when I'm done, read from buffer
        CALL_CL_GUARDED(clEnqueueReadBuffer, (queue, dev_buf_u, /*blocking*/ CL_TRUE, /*offset*/ 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
      }
     }
     resid2(curr, dxval[l]);
     injf2c(curr, curr->next); //this function updates f_{i+1}
     curr = curr->next;
  }
  // ----------------------------------------------------------------------
  // --------------- GPU ON THE COARSEST GRID -----------------------------
  // ----------------------------------------------------------------------
  {
    if(curr->dim_other < CUTOFF){
	for(i = 0; i < n3; i++){
	     gsrelax(curr, dxval[nl]);
	}
    }

    else{
  	// ---GPU------GPU------GPU------GPU------GPU------GPU------GPU------GPU--- //
  	// fill in the buffers inside the GPU with the current data
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_u, CL_TRUE, 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_f, CL_TRUE, 0, field_size * sizeof(ftype), curr->fvec, 0, NULL, NULL));
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_hist_u, CL_TRUE, 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
  	h = dxval[nl] * dxval[nl];
        size_t gdim[] = { curr->dim_x, curr->dim_x, curr->dim_x/z_div };
        size_t ldim[] = { wg_x, wg_y, wg_z };

  	for(i = 0; i < n3; i++){
     	   // ----------------------------------------------------------------------
     	   // invoke poisson kernel
     	   // ----------------------------------------------------------------------
	   curr->field_start = 0;
     	   SET_7_KERNEL_ARGS(poisson_knl, dev_buf_u, dev_buf_f, dev_buf_hist_u,curr->field_start, curr->dim_x, curr->dim_other, h);
     	   // run the kernel
	   curr->field_start = 0;
     	   CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, poisson_knl, /*dimensions*/ wg_dims, NULL, gdim, ldim, 0, NULL, evt_ptr));
     	   #ifdef DO_TIMING
      	   // If timing is enabled, this wait can mean a significant performance hit.
      	   CALL_CL_GUARDED(clWaitForEvents, (1, &evt));
 
      	   cl_ulong start, end;
      	   CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
      	   CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, CL_PROFILING_COMMAND_END, sizeof(start), &end, NULL));
 
      	   gbytes_accessed += 1e-9*(sizeof(ftype) * field_size * fetch_per_pt);
      	   seconds_taken += 1e-9*(end-start);
      	   mcells_updated += curr->dim_other*curr->dim_other*curr->dim_other/1e6;
      	   gflops_performed += 1e-9*curr->dim_x*curr->dim_x*curr->dim_x * flops_per_pt;

      	   CALL_CL_GUARDED(clReleaseEvent, (evt));
     	   #endif
      	   CALL_CL_GUARDED(clFinish, (queue)); //ira adentro??
	   cl_mem tmp = dev_buf_u;
	   dev_buf_u = dev_buf_hist_u;
	   dev_buf_hist_u = tmp;
     	}
        //when I'm done, read from buffer
        CALL_CL_GUARDED(clEnqueueReadBuffer, (queue, dev_buf_u, /*blocking*/ CL_TRUE, /*offset*/ 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
     }
  }
  // ----------------------------------------------------------------------
  // -----------Upward branch of the V-cycle ------------------------------
  // ----------------------------------------------------------------------
  for(l = nl-1; l >= 0; --l){
     ctof(curr->prev, curr, field_size); //curr->prev is the finer of the two
     free(curr->uvec);  //curr won't be needed anymore
     free(curr->fvec);
     free(curr->rvec);
     curr = curr->prev;
     curr->next = NULL;
     for(isweep = 0; isweep < n2; isweep++){
	   gsrelax(curr, dxval[l]);
     }
     // Update the grids n1 times using the GPU when necessary
     {
     if(curr->dim_other < CUTOFF){
	for(isweep = 0; isweep < n2; isweep++){
	     gsrelax(curr, dxval[l]);
	}
     }

     else{
  	// ---GPU------GPU------GPU------GPU------GPU------GPU------GPU------GPU--- //
  	// fill in the buffers inside the GPU with the current data
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_u, CL_TRUE, 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_f, CL_TRUE, 0, field_size * sizeof(ftype), curr->fvec, 0, NULL, NULL));
  	CALL_CL_GUARDED(clEnqueueWriteBuffer, (queue, dev_buf_hist_u, CL_TRUE, 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
  	h = dxval[l] * dxval[l];
        size_t gdim[] = { curr->dim_x, curr->dim_x, curr->dim_x/z_div };
        size_t ldim[] = { wg_x, wg_y, wg_z };

  	for(i = 0; i < n1; i++){
     	   // ----------------------------------------------------------------------
     	   // invoke poisson kernel
     	   // ----------------------------------------------------------------------
	   curr->field_start = 0;
     	   SET_7_KERNEL_ARGS(poisson_knl, dev_buf_u, dev_buf_f, dev_buf_hist_u, curr->field_start, curr->dim_x, curr->dim_other, h);
     	   // run the kernel
     	   CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, poisson_knl, /*dimensions*/ wg_dims, NULL, gdim, ldim, 0, NULL, evt_ptr));
     	   #ifdef DO_TIMING
      	   // If timing is enabled, this wait can mean a significant performance hit.
      	   CALL_CL_GUARDED(clWaitForEvents, (1, &evt));
 
      	   cl_ulong start, end;
      	   CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL));
      	   CALL_CL_GUARDED(clGetEventProfilingInfo, (evt, CL_PROFILING_COMMAND_END, sizeof(start), &end, NULL));
 
      	   gbytes_accessed += 1e-9*(sizeof(ftype) * field_size * fetch_per_pt);
      	   seconds_taken += 1e-9*(end-start);
	   tot_secs += 1e-9*(end-start_big);
      	   mcells_updated += curr->dim_other*curr->dim_other*curr->dim_other/1e6;
      	   gflops_performed += 1e-9*curr->dim_x*curr->dim_x*curr->dim_x * flops_per_pt;

      	   CALL_CL_GUARDED(clReleaseEvent, (evt));
     	   #endif
      	   CALL_CL_GUARDED(clFinish, (queue)); //ira adentro??
	   cl_mem tmp = dev_buf_u;
	   dev_buf_u = dev_buf_hist_u;
	   dev_buf_hist_u = tmp;
     	}
        //when I'm done, read from buffer
        CALL_CL_GUARDED(clEnqueueReadBuffer, (queue, dev_buf_u, /*blocking*/ CL_TRUE, /*offset*/ 0, field_size * sizeof(ftype), curr->uvec, 0, NULL, NULL));
      }
     }
  }
  // ---------- and the solution is right there in the last curr curr->uvec
  for(i = 0; i < field_size; i++)
     u[i] = curr->uvec[i];
  free(curr->uvec);
  //free(curr->fvec);
  free(curr->rvec);
  //free(ugrid->uvec);
  //free(ugrid->fvec);
  //free(ugrid->rvec);
  free(curr);
  CALL_CL_GUARDED(clReleaseMemObject, (dev_buf_u));
  CALL_CL_GUARDED(clReleaseMemObject, (dev_buf_f));
  CALL_CL_GUARDED(clReleaseMemObject, (dev_buf_hist_u));
}

void resid2(item * curr, ftype dx){
    
  size_t i, j, k;
  ftype h = dx * dx;
  unsigned base;
  for(i = 1; i < (curr->dim_other-1); i++)
     for(j = 1; j < (curr->dim_other-1); j++)
        for(k = 1; k < (curr->dim_other-1); k++){
           base = curr->field_start + i + curr->dim_x * (j + curr->dim_other * k);
           curr->rvec[base] = curr->fvec[base] +  ( curr->uvec[base+1] + curr->uvec[base-1] + 
	       curr->uvec[base - curr->dim_x] + curr->uvec[base + curr->dim_x] + 
	       curr->uvec[base - curr->dim_x * curr->dim_other] + 
	       curr->uvec[base + curr->dim_x * curr->dim_other] - 
	       6 * curr->uvec[base]) / h;
	}
	//I have doubts with regard to this formula
}

void injf2c(item * curr, item * futu ){
  // This functions transfers a fine grid to a coarse grid by injection
  size_t i, j, k;
  unsigned basef, basec;

  for(i = 0; i < futu->dim_other; i++)
     for(j = 0; j< futu->dim_other; j++)
        for(k = 0; k < futu->dim_other; k++){
	    basef = curr->field_start + (2*i) + curr->dim_x * ((2*j) + curr->dim_other * (2 * k));
  	    basec = futu->field_start + i + futu->dim_x * (j + futu->dim_other * k);
	    futu->fvec[basec] = curr->rvec[basef];  // notice here that I am injecting r in f.  Not f into f
	}
}

void ctof(item * finer, item * coarser, size_t field_size){
  // Transfers a coarse grid to a fine grid (bilinear interpolation)
  // Input:  finer and coarser grid structures
  // Output: finer grid gets updated
  unsigned base, basec, basef;
  size_t i,j,k;
  ftype *u; //malloc(sizeof(ftype) * max_size); can be created on the stack instead

  u = malloc(field_size * sizeof(ftype));
  for(i = 0; i < field_size; i++)
     u[i] = 0;
  // ----------- transfer by copying where the grids line up
  for(i = 0; i < coarser->dim_other; i++)
     for(j = 0; j < coarser->dim_other; j++)
        for(k = 0; k < coarser->dim_other; k++){
	   basef = finer->field_start + (2*i) + finer->dim_x * ((2*j) + finer->dim_other * (2 * k));
	   basec = coarser->field_start + i + coarser->dim_x * (j + coarser->dim_other * k);
	   u[basef] = coarser->uvec[basec];
	}

  // -----------    linear interpolation ----------------------------------

  // -----------    filling up lines --------------------------------------
  for(i = 1; i < finer->dim_other; i+=2)
     for(j = 2; j < finer->dim_other; j+=2)//can start in third row because first one is zero anyway
        for(k = 2; k < finer->dim_other; k+=2){
	   base = finer->field_start + i + finer->dim_x * (j + finer->dim_other * k);
	   u[base] = (u[base - 1] + u[base + 1])/2;
	}
  // ------------    filling complete even planes ---------------------------
    for(i = 0; i < finer->dim_other; i++) // notice that here it runs one by one
     for(j = 1; j < finer->dim_other; j+=2)//can start in third row because first one is zero anyway
        for(k = 2; k < finer->dim_other; k+=2){
	   base = finer->field_start + i + finer->dim_x * (j + finer->dim_other * k);
	   u[base] = (u[base - finer->dim_x] + u[base + finer->dim_x])/2;
	}
  	   
  // ------------    filling complete odd planes ---------------------------
    for(i = 0; i < finer->dim_other; i++) // notice that here it runs one by one
     for(j = 0; j < finer->dim_other; j++)//can start in third row because first one is zero anyway
        for(k = 1; k < finer->dim_other; k+=2){
	   base = finer->field_start + i + finer->dim_x * (j + finer->dim_other * k);
	   u[base] = (u[base - finer->dim_x * finer->dim_other] + u[base + finer->dim_x * finer->dim_other])/2;
	}  
  // ------------    Finally add this vector u to the corresponding grid ----
  //const int max_size  = POINTS * POINTS * ((POINTS + 15)/16) * 16;
  for(i = 0; i < field_size; i++)
	finer->uvec[i] += u[i];  
  free(u);
}

void gsrelax(item *curr, ftype dx){
  // This function does one sweep of Gauss-Seidel relaxation for the Poisson problem
  ftype h = dx*dx;
  size_t i,j,k;
  unsigned base;

  for(i = 1; i < (curr->dim_other-1); i++)
     for(j = 1; j < (curr->dim_other-1); j++)
  	for(k = 1; k < (curr->dim_other-1); k++){
	   base = curr->field_start + i + curr->dim_x * (j + curr->dim_other * k);
	   curr->uvec[base]=(curr->uvec[base-1] + curr->uvec[base+1] + curr->uvec[base - curr->dim_x] +	curr->uvec[base + curr->dim_x] + curr->uvec[base - curr->dim_x * curr->dim_other] + curr->uvec[base + curr->dim_x * curr->dim_other] + h * curr->fvec[base]) / 6;
	}
	   
}

ftype norm(ftype r[], unsigned field_size){
  // Calculates the "maximum" norm
  ftype nor = 0.0;
  for(size_t i = 0; i<field_size; i++){
     if(ABS(r[i]) > nor)
	nor = ABS(r[i]);
  }
  return(nor);
}
