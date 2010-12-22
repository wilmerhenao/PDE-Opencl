// workgroup: (8,8) fetch_per_pt:8 flops_per_pt:8
#pragma OPENCL EXTENSION cl_khr_fp64: enable
__kernel void fd_update(__global FTYPE *u, __global const FTYPE *f, __global const FTYPE *hist_u, const unsigned field_start, const unsigned dim_x, const unsigned dim_other, const FTYPE h)
{
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  unsigned base;

  if(i < (dim_other - 1) && j < (dim_other-1) && i > 0 && j > 0){ //enter the for cycle only when necessary --- otherwise do nothing

    for (int k = 1; k < (dim_other-1); ++k){
      base = field_start + i + dim_x*(j + dim_other * k);
      u[base] = (
           hist_u[base-1] + hist_u[base+1] 
           + hist_u[base - dim_x] + hist_u[base + dim_x] 
           + hist_u[base - dim_x * dim_other] + hist_u[base + dim_x * dim_other] 
           /*+ h * f[base]*/
           ) / 6;
    }
  }
}
