/* ---------------------------------------------------------------------

   solve 3d poisson equation   lapl u = f  u = g on bndry
   in square domain  [xlo, xhi] by [ylo, yhi] by [zlo, zhi] (set to [-1, 1])
   with dirichlet boundary conditions using simple jacobi iteration (GS or SOR
   are optional - see comments in code for how to turn on).

-------------------------------------------------------------------- */

# include <stdlib.h>
# include <stdio.h>
# include <math.h>

#include <time.h>
#include "timing.h"

/* ----- local prototypes -------------------------------------------------- */

int main ( int argc, char *argv[] );
void get_prob_size(int *nx, int *ny, int *nz, int argc, char** argv);
void driver ( int nx, int ny, int nz, int it_max, double tol,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval );
void jacobi ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval );
void init_prob ( int nx, int ny, int nz, double f[], double u[],
                 double xlo, double ylo, double zlo,
                 double xhi, double yhi, double zhi );
void calc_error ( int nx, int ny, int nz, double u[], double f[],
                  double xlo, double ylo, double zlo,
                  double xhi, double yhi, double zhi  );
double u_exact   ( double x, double y, double z );
double uxx_exact ( double x, double y, double z );
double uyy_exact ( double x, double y, double z );
double uzz_exact ( double x, double y, double z );


/* ----- local macros  -------------------------------------------------- */

#define INDEX3D(i,j,k)    ((i)+(j)*nx + (k)*(nx)*(ny))  //NOTE: column major (fortran-style) ordering here
#define MAX(A,B)          ((A) > (B)) ? (A) : (B)
#define ABS(A)            ((A) >= 0) ? (A) : -(A)
#define F(i,j,k)          f[INDEX3D(i,j,k)]
#define U(i,j,k)          u[INDEX3D(i,j,k)]
#define U_OLD(i,j,k)      u_old[INDEX3D(i,j,k)]

/* ------------------------------------------------------------------------- */

int main ( int argc, char *argv[] )
{
  
    int nx = -1;   /* number of grid points in x           */
    int ny = -1;   /*                     and  y direction */
    int nz = -1;   /*                     and  z direction */

    double tol = 1.e-7;     /* convergence criteria */
    int it_max = 1000;      /* max number of iterations */
    int io_interval =  100; /* output status this often */


    double xlo = -1., ylo = -1., zlo = -1; /* lower corner of domain */
    double xhi =  1., yhi =  1., zhi = 1.; /* upper corner of domain */
  
    
    /* get number of grid points for this experiment */
    get_prob_size(&nx, &ny, &nz, argc, argv);
  
    //wval1 = omp_get_wtime();
    
    driver ( nx, ny, nz, it_max, tol, xlo, ylo, zlo, xhi, yhi, zhi, io_interval );

    //wval2 = omp_get_wtime();
    //printf("omp walltime  = %15.7e\n",wval2-wval1);
    
    
    return 0;
}

/* ------------------------------------------------------------------------- */

void driver ( int nx, int ny, int nz, int it_max, double tol,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval)
{
    double *f, *u;
    int i,j,k;
    double secs = -1.0;
    struct timespec start, finish;


    /* Allocate and initialize  */
    f = ( double * ) malloc ( nx * ny * nz * sizeof ( double ) );  
    u = ( double * ) malloc ( nx * ny * nz * sizeof ( double ) );

    get_time(&start);

    for ( k = 0; k < nz; k++ ) 
      for ( j = 0; j < ny; j++ ) 
        for ( i = 0; i < nx; i++ ) 
          U(i,j,k) = 0.0;  /* note use of array indexing macro */


  /* set rhs, and exact bcs in u */
    init_prob ( nx, ny, nz, f, u , xlo, ylo, zlo, xhi, yhi, zhi);

    /* Solve the Poisson equation  */
    jacobi ( nx, ny, nz, u, f, tol, it_max, xlo, ylo, zlo, xhi, yhi, zhi, io_interval  );

    /* Determine the error  */
    calc_error ( nx, ny, nz, u, f, xlo, ylo, zlo, xhi, yhi, zhi );

    /* get time for initialization and iteration.  */
    get_time(&finish);
    secs = timespec_diff(start,finish);
    printf(" Total time: %15.7e seconds\n",secs);

    free ( u );
    free ( f );

    return;
}


/* ------------------------------------------------------------------------- */


void jacobi ( int nx, int ny, int nz, double u[], double f[], double tol, int it_max,
              double xlo, double ylo, double zlo,
              double xhi, double yhi, double zhi, int io_interval)
{

    double ax, ay, az, d;
    double dx, dy, dz;
    double update_norm, unew;
    int i, it, j, k;
    double *u_old, diff;

    /* Initialize the coefficients.  */

    dx =  (xhi - xlo) / ( double ) ( nx - 1 );
    dy =  (yhi - ylo) / ( double ) ( ny - 1 );
    dz =  (zhi - zlo) / ( double ) ( nz - 1 );

    ax =   1.0 / (dx * dx);
    ay =   1.0 / (dy * dy);
    az =   1.0 / (dz * dz);
    d  = - 2.0 / (dx * dx)  - 2.0 / (dy * dy) -2.0 / (dz * dz);

    u_old = ( double * ) malloc ( nx * ny * nz * sizeof ( double ) );

    for ( it = 1; it <= it_max; it++ ) {
        update_norm = 0.0;

        /* Copy new solution into old.  */
      for ( k = 0; k < nz; k++ ) {
        for ( j = 0; j < ny; j++ ) {
            for ( i = 0; i < nx; i++ ) {
              U_OLD(i,j,k) = U(i,j,k);
            }
        }
      }

    /* Compute stencil, and update.  bcs already in u. only update interior of domain */
      for ( k = 1; k < nz-1; k++ ) {
        for ( j = 1; j < ny-1; j++ ) {
            for ( i = 1; i < nx-1; i++ ) {

                unew = (F(i,j,k) -
                        ( ax * ( U_OLD(i-1,j,k) + U_OLD(i+1,j,k) ) +
                          ay * ( U_OLD(i,j-1,k) + U_OLD(i,j+1,k) ) +
                          az * ( U_OLD(i,j,k-1) + U_OLD(i,j,k+1) ) ) ) / d;

                diff = ABS(unew-U_OLD(i,j,k));
                update_norm = update_norm + diff*diff;  /* using 2 norm */

                if (diff > update_norm){ /* using max norm */
                    update_norm = diff;
                  }

                U(i,j,k) = unew;

            } /* end for i */
        } /* end for j */
      } /* end for k */

        if (0 == it% io_interval) 
            printf ( " iteration  %5d   norm update %12.3e\n", it, update_norm );

        if ( update_norm <= tol ) {
          break;
        }

    } /* end for it iterations */

    printf ( " iteration  %5d   norm update %12.3e\n", it, update_norm );
    printf ( "  Total number of iterations %d\n", it );

    free ( u_old );

    return;
}

/******************************************************************************/

void init_prob ( int nx, int ny, int nz, double  f[], double u[],
                 double xlo,  double ylo, double zlo,
                 double xhi,  double yhi, double zhi )
{
  int    i, j, k;
  double x, y, z, dx, dy, dz;

    dx = (xhi - xlo) / ( double ) ( nx - 1 );
    dy = (yhi - ylo) / ( double ) ( ny - 1 );
    dz = (zhi - zlo) / ( double ) ( nz - 1 );

    /* Set the boundary conditions.  */
    j = 0;   // low y bc
    y = ylo + j * dy;

    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dz;
      for ( i = 0; i < nx; i++ ) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    j = ny - 1;  // hi y
    y = ylo + j * dy;
    
    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dz;      
      for ( i = 0; i < nx; i++ ) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    i = 0;  // low x
    x = xlo + i * dx;

    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dx;
      for ( j = 0; j < ny; j++ ) {
        y = ylo + j * dy;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    i = nx - 1; // hi x
    x = xlo + i * dx;

    for ( k = 0; k < nz; k++ ) {
      z = zlo + k*dx;    
      for ( j = 0; j < ny; j++ ) {
        y = ylo + j * dy;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    k = 0; // low z
    z = zlo + k * dz;
    
    for ( j = 0; j < ny; j++ ) {
      y = ylo + j * dy;
      for ( i = 0; i < nx; i++) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }

    k = nz - 1; // hi z
    z = zlo + k * dz;
    
    for ( j = 0; j < ny; j++ ) {
      y = ylo + j * dy;
      for ( i = 0; i < nx; i++) {
        x = xlo + i * dx;
        U(i,j,k) = u_exact ( x, y, z );
      }
    }



    /* Set the right hand side  */
    for ( k = 0; k < nz; k++ ){
      z = zlo + k * dz;
      for ( j = 0; j < ny; j++ ){
        y = ylo + j * dy;
        for ( i = 0; i < nx; i++ ) {
          x = xlo + i * dx;
          F(i,j,k) =  uxx_exact ( x, y, z ) + uyy_exact ( x, y, z ) + uzz_exact( x, y, z );
        }
      }
    }

    return ;
}

/* ------------------------------------------------------------------------- */

void calc_error ( int nx, int ny, int nz, double u[],  double f[],
                  double xlo,  double ylo, double zlo,
                  double xhi,  double yhi, double zhi )
{
    double  error_max, error_l2;
    int     i, j, k, i_max=-1, j_max=-1, k_max = -1;
    double  u_true, u_true_norm;
    double  x, y, z, dx, dy, dz, term;

    dx = (xhi - xlo) / ( double ) ( nx - 1 );
    dy = (yhi - ylo) / ( double ) ( ny - 1 );
    dz = (yhi - ylo) / ( double ) ( nz - 1 );

    error_max   = 0.0;
    error_l2    = 0.0;

    /* print statements below are commented out but  may help in debugging */
    //printf("   i    j   k       x       y    z         uexact          ucomp            error\n");
    
    for ( k = 0; k < nz; k++ ) {
      z = zlo + k * dz;
      for ( j = 0; j < ny; j++ ) {
        y = ylo + j * dy;
        for ( i = 0; i < nx; i++ ) {
            x = xlo + i * dx;
            u_true = u_exact ( x, y, z );
            term   =  U(i,j,k) - u_true;
            //printf(" %d  %d  %d %12.5e %12.5e %12.5e %12.5e  %12.5e %12.5e \n",i,j,k,x,y,z,u_true,U(i,j,k),term);
            error_l2  = error_l2 + term*term;
            if (ABS(term) > error_max){
              error_max =  ABS(term);
            }
        } /* end for i */
      } /* end for j */
    } /* end for k */

    error_l2 = sqrt(dx*dy*dz*error_l2);

    printf ( "\n  max Error in computed soln:    %12.5e    \n", error_max );
    printf ( "  l2 norm of Error on %4d by %4d by %4d grid\n  (dx %12.5e dy %12.5e dz %12.5e):   %12.5e\n",
             nx,ny,nz,dx,dy,dz,error_l2 );

    return;
}

/* ------------------------------------------------------------------------- */

double u_exact ( double x, double y, double z )
{
  double pi = 4.*atan(1.0);

  return (( 1.0 - x * x ) * ( 1.0 - y * y ) * (1.0 - z * z));  // exact solution
  //return (cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return (cos(pi*x) + cos(pi*y) + cos(pi*z));
  //return 2.;
}

/* ------------------------------------------------------------------------- */

double uxx_exact ( double x, double y, double z )
{
  double pi = 4.*atan(1.0);
  
  return (-2.0 * ( 1.0 + y ) * ( 1.0 - y ) * (1.0 - z) * (1.0 + z));
  //return (-pi*pi*cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return(-pi*pi*cos(pi*x));
  //return 0;
}

/* ------------------------------------------------------------------------- */

double uyy_exact ( double x, double y , double z)
{
  double pi = 4.*atan(1.0);
  
  return (-2.0 * ( 1.0 + x ) * ( 1.0 - x ) * (1.0 - z) * (1.0 + z));
  //return (-pi*pi*cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return(-pi*pi*cos(pi*y));
  //return 0;
}

/* ------------------------------------------------------------------------- */

double uzz_exact ( double x, double y , double z)
{
  double pi = 4.*atan(1.0);
  
  return (-2.0 * ( 1.0 + x ) * ( 1.0 - x ) * (1.0 - y) * (1.0 + y));
  //return (-pi*pi*cos(pi*x)*cos(pi*y)*cos(pi*z));
  //return(-pi*pi*cos(pi*z));
  //return 0;
}

/* ------------------------------------------------------------------------- */

void get_prob_size(int *nx, int *ny, int *nz, int argc, char** argv)
{

    if (4 == argc) {
        /* read problem size from the arguments */
        *nx  = atoi(argv[1]);
        *ny  = atoi(argv[2]);
        *nz  = atoi(argv[3]);
    } else {
        /* input problem size for this experiment */
        printf("Enter 3 integers for number of grid points in x y and z\n");
        scanf("%d %d %d",nx,ny,nz);
    }

    if (0 >= *nx  || 0 >= *ny || 0 >= *nz ) {
      printf(" problem with domain discretizations nx = %d n = %d nz = %d\n", *nx,*ny,*nz);
        exit(-11);
    } else {
      printf("Discretizing with %d %d %d points in x y and z \n", *nx, *ny, *nz);
    }

    return;
}

/* ------------------------------------------------------------------------- */

#undef U_OLD
#undef U
#undef F
#undef ABS
#undef MAX
#undef INDEX3D
