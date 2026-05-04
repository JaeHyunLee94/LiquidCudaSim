//
// Created by test on 2022-09-28.
//

#ifndef MPM_SOLVER_SRC_ENGINE_CUDA_CUDATYPES_CUH_
#define MPM_SOLVER_SRC_ENGINE_CUDA_CUDATYPES_CUH_

#include "nvfunctional"
#include "../Types.h"
namespace mpm{

struct float9{

  Scalar data[9];

  __forceinline__ __device__ Scalar &operator[](int i) {
    return data[i];
  }
  __forceinline__ __device__ const Scalar &operator[](int i) const {
    return data[i];
  }

  };


using StressFunc = nvstd::function<void( float9&, Scalar&,float9&)>;
using ProjectFunc= nvstd::function<void(float9& ,float9&,Scalar&, Scalar)> ;
//using ParticleConstraintFunc = std::function<void(int , Scalar*,Scalar*)>;
//using GridConstraintFunc = nvstd::function<void(int,int,int,Scalar*,Scalar*)>;
//TODO: index operator overloading

}

#endif //MPM_SOLVER_SRC_ENGINE_CUDA_CUDATYPES_CUH_
