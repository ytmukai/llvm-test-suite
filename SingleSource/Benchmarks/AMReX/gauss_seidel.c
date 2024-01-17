#include <stdlib.h>

// Modified from https://github.com/AMReX-Codes/amrex/blob/development/Src/LinearSolvers/MLMG/AMReX_MLNodeLap_2D_K.H
// The kernel is simplified to allow optimization easier.
// The number of iterations is set to hit L1 cache.
// Performance characteristics may not reflect the real application.

typedef double Real;

__attribute__((noinline))
void mlndlap_gauss_seidel_aa(int nx, int ny, int nz,
            Real sol[restrict][ny][nx],
            Real sol2[restrict][ny][nx],
            Real sig[restrict][ny][nx],
            Real rhs[restrict][ny][nx],
            Real fxy, Real f2xmy, Real fmx2y, Real facy) {
  for (int i = 1; i < nz-1; ++i) {
    for (int j = 1; j < ny-1; ++j) {
      for (int k = 0; k < nx; ++k) {
          Real s0 = (-(Real)(2.0))*fxy*(sig[i-1][j-1][k]+sig[i][j-1][k]+sig[i-1][j][k]+sig[i][j][k]);
          Real Ax =   sol[i-1][j-1][k]*fxy*sig[i-1][j-1][k]
            + sol[i+1][j-1][k]*fxy*sig[i  ][j-1][k]
            + sol[i-1][j+1][k]*fxy*sig[i-1][j  ][k]
            + sol[i+1][j+1][k]*fxy*sig[i  ][j  ][k]
            + sol[i-1][j][k]*f2xmy*(sig[i-1][j-1][k]+sig[i-1][j][k])
            + sol[i+1][j][k]*f2xmy*(sig[i  ][j-1][k]+sig[i  ][j][k])
            + sol[i][j-1][k]*fmx2y*(sig[i-1][j-1][k]+sig[i][j-1][k])
            + sol[i][j+1][k]*fmx2y*(sig[i-1][j  ][k]+sig[i][j  ][k])
            + sol[i][j][k]*s0;

            Real fp = facy / (Real)(2*i+1);
            Real fm = facy / (Real)(2*i-1);
            Real frzlo = fm*sig[i-1][j-1][k]-fp*sig[i][j-1][k];
            Real frzhi = fm*sig[i-1][j  ][k]-fp*sig[i][j  ][k];
            s0 += - frzhi - frzlo;
            Ax += frzhi*(sol[i][j+1][k]-sol[i][j][k])
              + frzlo*(sol[i][j-1][k]-sol[i][j][k]);

          sol2[i][j][k] += (rhs[i][j][k] - Ax) / s0;
      }
    }
  }
}

// Original kernel:
//
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// void mlndlap_gauss_seidel_aa (Box const& bx, Array4<Real> const& sol,
//                               Array4<Real const> const& rhs, Array4<Real const> const& sig,
//                               Array4<int const> const& msk,
//                               GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
//                               bool is_rz) noexcept
// {
//     Real facx = Real(1.0/6.0)*dxinv[0]*dxinv[0];
//     Real facy = Real(1.0/6.0)*dxinv[1]*dxinv[1];
//     Real fxy = facx + facy;
//     Real f2xmy = Real(2.0)*facx - facy;
//     Real fmx2y = Real(2.0)*facy - facx;
// 
//     amrex::Loop(bx, [=] (int i, int j, int k) noexcept
//     {
//         if (msk(i,j,k)) {
//             sol(i,j,k) = Real(0.0);
//         } else {
//             Real s0 = (-Real(2.0))*fxy*(sig(i-1,j-1,k)+sig(i,j-1,k)+sig(i-1,j,k)+sig(i,j,k));
//             Real Ax =   sol(i-1,j-1,k)*fxy*sig(i-1,j-1,k)
//                       + sol(i+1,j-1,k)*fxy*sig(i  ,j-1,k)
//                       + sol(i-1,j+1,k)*fxy*sig(i-1,j  ,k)
//                       + sol(i+1,j+1,k)*fxy*sig(i  ,j  ,k)
//                       + sol(i-1,j,k)*f2xmy*(sig(i-1,j-1,k)+sig(i-1,j,k))
//                       + sol(i+1,j,k)*f2xmy*(sig(i  ,j-1,k)+sig(i  ,j,k))
//                       + sol(i,j-1,k)*fmx2y*(sig(i-1,j-1,k)+sig(i,j-1,k))
//                       + sol(i,j+1,k)*fmx2y*(sig(i-1,j  ,k)+sig(i,j  ,k))
//                       + sol(i,j,k)*s0;
// 
//             if (is_rz) {
//                 Real fp = facy / static_cast<Real>(2*i+1);
//                 Real fm = facy / static_cast<Real>(2*i-1);
//                 Real frzlo = fm*sig(i-1,j-1,k)-fp*sig(i,j-1,k);
//                 Real frzhi = fm*sig(i-1,j  ,k)-fp*sig(i,j  ,k);
//                 s0 += - frzhi - frzlo;
//                 Ax += frzhi*(sol(i,j+1,k)-sol(i,j,k))
//                     + frzlo*(sol(i,j-1,k)-sol(i,j,k));
//             }
// 
//             sol(i,j,k) += (rhs(i,j,k) - Ax) / s0;
//         }
//     });
// }

int main(int argc, char *argv[]) {
  const int NX = 100, NY = 3, NZ = 3;
  const int Rep = 1000000;
  
  typedef Real (*ArrType)[NY][NX];
  void *buf;
  posix_memalign(&buf, 1024, sizeof(Real)*NX*NY*NZ*4);
  ArrType sol =  (ArrType)buf + NZ*0;
  ArrType sig =  (ArrType)buf + NZ*1;
  ArrType rhs =  (ArrType)buf + NZ*2;
  ArrType sol2 = (ArrType)buf + NZ*3;

  for (int i=0; i<NZ; ++i)
    for (int j=0; j<NY; ++j)
      for (int k=0; k<NX; ++k)
        sol[i][j][k] = sig[i][j][k] = rhs[i][j][k] = sol2[i][j][k] = i+j+k;
  
  for (int i=0; i<Rep; i++)
    mlndlap_gauss_seidel_aa(NX, NY, NZ, sol, sol2, sig, rhs, 1, 2, 3, 4);
  
  return 0;
}
