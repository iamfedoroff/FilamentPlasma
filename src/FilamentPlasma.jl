module FilamentPlasma

import Adapt: @adapt_structure
import CUDA: @cuda, @captured, launch_configuration, CuArray, threadIdx,
             blockIdx, blockDim, gridDim
import DelimitedFiles: readdlm
import ODEIntegrators: Problem, Integrator, step, step!, RK2, RK3, SSPRK3,
                       SSP4RK3, RK4, Tsit5, ATsit5

using PhysicalConstants.CODATA2018
const QE = ElementaryCharge.val
const ME = ElectronMass.val
const HBAR = ReducedPlanckConstant.val

import FilamentBase: @krun, linterp, Grid, GridRT

# ODEIntegrators
export RK2, RK3, SSPRK3, SSP4RK3, RK4, Tsit5, ATsit5

# export PT1, PT2, PT3, PT4, PT5
export PT1, PT4, PT5, IonizationRate


abstract type PType end

abstract type Plasma end


include("plasma_pt1.jl")
# include("plasma_pt2.jl")
# include("plasma_pt3.jl")
include("plasma_pt4.jl")
include("plasma_pt5.jl")

include("ionrates.jl")


"""
In general, arrays of functions are not supported inside CUDA kernels [1].
However, this function allows to circumvent the issue [2].
[1] https://discourse.julialang.org/t/cuda-kernel-how-to-pass-an-array-of-functions
[2] https://discourse.julialang.org/t/optimizing-cuda-jl-performance-for-small-array-operations/54808/4

A limitation of this approach is that it stops working once the number of
functions is larger than 10 because of
https://github.com/JuliaLang/julia/blob/5ab3ed6415bacb2fd4dd41302bde010c46074c2f/base/ntuple.jl#L17-L32
"""
apply_funcs(fs, x) = ntuple(i -> fs[i](x), length(fs))



end
