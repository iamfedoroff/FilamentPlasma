# ******************************************************************************
# CUDA version is not implemented due to CuArray of CuArrays issue:
# https://discourse.julialang.org/t/arrays-of-arrays-and-arrays-of-structures-in-cuda-kernels-cause-random-errors
#
# function func!(du, u, p, t, E)        function solve!(plasma, t, E)
#     for ic=1:Ncomp                        for ir=1:Nr
#         du[ic] = ...                          integ = integs[ir]
#     end                                       @. utmp = integ.prob.u0
#     return nothing                            for it=1:Nt-1
# end                                               step!(integ, utmp, t[it], dt, E[ir,:])
#                                                   @. u[ir,it+1,:] = utmp
#                                               end
#                                           end
#                                           return nothing
#                                       end
# ******************************************************************************

struct PT3 <: PType end


struct PlasmaPT3{T, G, TI, TN, TU}
    Nt :: Int
    dt :: T
    t :: G
    integs :: TI
    neu :: T
    ne :: TN
    u :: TU
end

@adapt_structure PlasmaPT3


function Plasma(
    grid::GridRT, field, medium, type::PT3; components, alg=RK3(), neu=nothing,
)
    (; Nr, Nt, tu, t, dt) = grid
    (; Iu) = field
    (; N0) = medium

    if isnothing(neu)
        neu = N0
    end

    Ncomp = length(components)
    fracs = zeros(Ncomp)
    ionrates = Array{Function}(undef, Ncomp)
    for (i, comp) in enumerate(components)
        fracs[i] = comp["frac"]
        ionrates[i] = comp["ionrate"]
    end
    N0s = @. fracs * N0 / neu
    ionrates = Tuple(ionrates)   # to avoid allocations and prevent Arrays adaptor

    ne = zeros(Nr, Nt)
    u = zeros(Ncomp)

    prob = Problem(func_pt3!, u, (tu, Iu, ionrates, N0s, t))
    integs = Array{Integrator}(undef, Nr)
    for ir=1:Nr
        integs[ir] = Integrator(prob, alg)
    end
    integs = Tuple(integs)   # to avoid allocations and prevent Arrays adaptor

    return PlasmaPT3(Nt, dt, t, integs, neu, ne, u)
end


function func_pt3!(du, u, p, t, EE)
    tu, Iu, ionrates, N0s, tt = p
    Ncomp = length(u)
    E = linterp(t, tt, EE)
    I = abs2(E) * Iu
    Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
    for ic=1:Ncomp
        du[ic] = Rs[ic]*tu * (N0s[ic] - u[ic])
    end
    return nothing
end


function solve!(plasma::PlasmaPT3, E)
    (; dt, t, integs, ne, u) = plasma
    Nr, Nt = size(E)
    for ir=1:Nr
        integ = integs[ir]
        @. u = integ.prob.u0
        ne[ir,1] = sum(u)
        for it=1:Nt-1
            @views step!(integ, u, t[it], dt, E[ir,:])
            ne[ir,it+1] = sum(u)
        end
    end
    return nothing
end


function solve!(plasma::PlasmaPT3, E::CuArray)
    @error "CUDA version is not implemented due to CuArray of CuArrays issue."
    # Nr = size(E,1)
    # @krun Nr solve_pt3_kernel!(u, utmp, t, E, integs)
    return nothing
end
# function solve_pt3_kernel!(u, utmp, t, E, integs)
#     id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = blockDim().x * gridDim().x

#     Nr, Nt = size(E)
#     dt = t[2] - t[1]
#     for ir=id:stride:Nr
#         integ = integs[ir]
#         @. utmp = integ.prob.u0
#         @. u[:,ir,1] = utmp
#         for it=1:Nt-1
#             @views step!(integ, utmp, t[it], dt, E[ir,:])
#             @. u[:,ir,it+1] = utmp
#         end
#     end
#     return nothing
# end
