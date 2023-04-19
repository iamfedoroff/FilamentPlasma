# ******************************************************************************
# function func!(du, u, p, t, E)        function solve!(u, utmp, integ, t, E)
#     for ir=1:Nr                           @. utmp = integ.prob.u0
#         for ic=1:Ncomp                    @. u[:,1,:] = utmp
#             du[ir,ic] = ...               for it=1:Nt-1
#         end                                   step!(integ, utmp, t[it], dt, E)
#     end                                       @. u[:,it+1,:] = utmp
#     return nothing                        end
# end                                       return nothing
#                                       end
# ******************************************************************************

struct PT1 <: PType end


struct PlasmaPT1{T, G, TI, TU, TN} <: Plasma
    Nt :: Int
    neu :: T
    t :: G
    dt :: T
    integ :: TI
    u :: TU
    ne :: TN
    kdne :: TN
end

@adapt_structure PlasmaPT1


function Plasma(
    grid::GridRT, field, medium, type::PT1; components, alg=RK3(), neu=1,
)
    (; Nr, Nt, tu, t, dt) = grid
    (; Iu) = field
    (; N0) = medium

    Ncomp = length(components)
    fracs = zeros(Ncomp)
    ionrates = Array{Function}(undef, Ncomp)
    for (i, comp) in enumerate(components)
        fracs[i] = comp["frac"]
        ionrates[i] = comp["ionrate"]
    end
    N0s = @. fracs * N0 / neu
    ionrates = Tuple(ionrates)   # to avoid allocations and prevent Arrays adaptor

    u = zeros(Nr, Ncomp)
    ne = zeros(Nr, Nt)
    kdne = zeros(Nr, Nt)

    prob = Problem(func_pt1!, u, (tu, Iu, ionrates, N0s, t))
    integ = Integrator(prob, alg)

    return PlasmaPT1(Nt, neu, t, dt, integ, u, ne, kdne)
end


function func_pt1!(du, u, p, t, EE)
    tu, Iu, ionrates, N0s, tt = p
    Nr, Ncomp = size(u)
    for ir=1:Nr
        E = @views linterp(t, tt, EE[ir,:])
        I = abs2(E) * Iu
        Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
        for ic=1:Ncomp
            du[ir,ic] = Rs[ic]*tu * (N0s[ic] - u[ir,ic])
        end
    end
    return nothing
end


function func_pt1!(du, u, p, t, EE::CuArray)
    Nr = size(EE, 1)
    @krun Nr func_pt1_kernel!(du, u, p, t, EE)
end
function func_pt1_kernel!(du, u, p, t, EE)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    tu, Iu, ionrates, N0s, tt = p
    Nr, Ncomp = size(u)
    for ir=id:stride:Nr
        E = @views linterp(t, tt, EE[ir,:])
        I = abs2(E) * Iu
        Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
        for ic=1:Ncomp
            du[ir,ic] = Rs[ic]*tu * (N0s[ic] - u[ir,ic])
        end
    end
    return nothing
end


function solve!(plasma::PlasmaPT1, E)
    (; Nt, dt, t, integ, ne, u) = plasma
    @. u = integ.prob.u0
    @views sum!(ne[:,1], u)
    for it=1:Nt-1
        step!(integ, u, t[it], dt, E)
        @views sum!(ne[:,it+1], u)
    end
    return nothing
end
