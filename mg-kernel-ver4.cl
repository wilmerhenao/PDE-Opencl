// workgroup: (8,8) fetch_per_pt:5 flops_per_pt:8
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define WG_SIZE 16
__kernel void fd_update(__global FTYPE *u, __global const FTYPE *f, __global const FTYPE *hist_u, const unsigned field_start, const unsigned dim_x, const unsigned dim_other, const FTYPE h)
{
  __local FTYPE lu[WG_SIZE+2][WG_SIZE+2];

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int li = get_local_id(0) + 1;
  const int lj = get_local_id(1) + 1;
  unsigned base;

    for (int k = 1; k < (dim_other-1); ++k){
      base = field_start + i + dim_x*(j + dim_other * k);
      lu[li][lj] = hist_u[base];

      if(1 == li){
	lu[0][lj] = hist_u[base-1];
   	lu[1+WG_SIZE][lj] = hist_u[base + WG_SIZE];
      }

      if(1 == lj){
	lu[li][0] = hist_u[base - dim_x];
      	lu[li][1 + WG_SIZE] = hist_u[base + WG_SIZE * dim_x];
      }

      barrier(CLK_LOCAL_MEM_FENCE);

      if(i < (dim_other - 1) && j < (dim_other-1) && i > 0 and j > 0)
	 u[base]=(lu[li-1][lj] + lu[li+1][lj] + lu[li][lj-1] + lu[li][lj+1] + hist_u[base - dim_x * dim_other] + hist_u[base + dim_x * dim_other] + h * f[base]) / 6;

      barrier(CLK_LOCAL_MEM_FENCE);
      
   }
}
