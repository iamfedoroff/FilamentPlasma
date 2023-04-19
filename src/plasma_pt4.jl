# ******************************************************************************
# For CUDA needs additional conversion of integrators:
#    integs = CuArray(hcat([integs[i] for i in 1:Nr]))
#
# function func!(u, p, t, E)                function solve!(plasma, t, E)
#     ...                                       ...
#     du = SVector{Ncomp}(u)                    for ir=1:Nr
#     for ic=1:Ncomp                                integ = integs[ir]
#         tmp = ...                                 utmp = integ.prob.u0
#         du = setindex(du, tmp, ic)                @. u[ir,1,:] = utmp
#     end                                           for it=1:Nt-1
#     return du                                         utmp = step(integ, utmp, t[it], dt, E[ir,:])
# end                                                   @. u[ir,it+1,:] = utmp
#                                                   end
#                                                end
#                                                return nothing
#                                            end
# ******************************************************************************

import StaticArrays: SVector, setindex


struct PT4 <: PType end


struct PlasmaPT4{T, G, TF, TA, TI1, TI2, TN}
    Ncomp :: Int
    Nt :: Int
    tu :: T
    Iu :: T
    neu :: T
    t :: G
    dt :: T
    mr :: T
    nuc :: T
    ionrates :: TF
    N0s :: TA
    Ks :: TA
    integs :: TI1
    integs_pi :: TI2
    ne :: TN
    kdne :: TN
end

@adapt_structure PlasmaPT4


function Plasma(
    grid::GridRT, field, medium, type::PT4;
    components, alg=RK3(), neu=1, mr=1, nuc=0,
)
    (; Nr, Nt, tu, t, dt) = grid
    (; Eu, Iu, w0) = field
    (; N0) = medium

    Ncomp = length(components)
    N0s, Ravas, Ks = (zeros(Ncomp) for i=1:3)
    ionrates = Array{Function}(undef, Ncomp)
    for (i, comp) in enumerate(components)
        N0s[i] = comp["frac"] * N0 / neu

        ionrates[i] = comp["ionrate"]

        Ui = comp["Ui"] * QE   # eV -> J
        MR = mr * ME
        sigmaB = QE^2 / MR * nuc / (nuc^2 + w0^2)
        Ravas[i] = sigmaB / Ui * tu * Eu^2

        Ks[i] = ceil(Ui / (HBAR * w0))
    end
    N0s, Ravas, Ks = (SVector{Ncomp}(x) for x in (N0s, Ravas, Ks))
    ionrates = Tuple(ionrates)   # to avoid allocations and prevent Arrays adaptor

    ne = zeros(Nr, Nt)
    kdne = zeros(Nr, Nt)

    tu, Iu, neu, dt, mr, nuc = promote(tu, Iu, neu, dt, mr, nuc)

    u0 = SVector{Ncomp}(zeros(Ncomp))
    prob = Problem(func_pt4, u0, (tu, Iu, ionrates, N0s, Ravas, t))
    prob_pi = Problem(func_pt4_pi, u0, (tu, Iu, ionrates, N0s, t))
    integs = Array{Integrator}(undef, Nr)
    integs_pi = Array{Integrator}(undef, Nr)
    for ir=1:Nr
        integs[ir] = Integrator(prob, alg)
        integs_pi[ir] = Integrator(prob_pi, alg)
    end
    integs = Tuple(integs)   # to avoid allocations and prevent Arrays adaptor
    integs_pi = Tuple(integs_pi)   # to avoid allocations and prevent Arrays adaptor

    return PlasmaPT4(
        Ncomp, Nt, tu, Iu, neu, t, dt, mr, nuc, ionrates, N0s, Ks, integs,
        integs_pi, ne, kdne,
    )
end


function func_pt4(u, p, t, EE)
    tu, Iu, ionrates, N0s, Ravas, tt = p
    Ncomp = length(u)

    E = linterp(t, tt, EE)
    I = abs2(E) * Iu
    E2 = real(E)^2

    Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
    utot = sum(u)   # total electron density

    du = SVector{Ncomp}(u)
    for ic=1:Ncomp
        N0 = N0s[ic]
        R1 = Rs[ic] * tu
        R2 = Ravas[ic] * E2
        tmp = R1 * (N0 - u[ic]) + R2 * (N0 - u[ic]) / N0 * utot
        du = setindex(du, tmp, ic)
    end
    return du
end


function func_pt4_pi(u, p, t, EE)
    tu, Iu, ionrates, N0s, tt = p
    Ncomp = length(u)

    E = linterp(t, tt, EE)
    I = abs2(E) * Iu

    Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs

    du = SVector{Ncomp}(u)
    for ic=1:Ncomp
        N0 = N0s[ic]
        R1 = Rs[ic] * tu
        tmp = R1 * (N0 - u[ic])
        du = setindex(du, tmp, ic)
    end
    return du
end


function solve!(plasma::PlasmaPT4, E)
    (; tu, Iu, t, dt, ionrates, N0s, Ks, integs, integs_pi, ne, kdne) = plasma
    @. kdne = 0
    Nr, Nt = size(E)
    for ir=1:Nr
        integ = integs[ir]
        integ_pi = integs_pi[ir]
        u = integ.prob.u0
        u_pi = integ_pi.prob.u0
        ne[ir,1] = sum(u)
        for it=1:Nt-1
            u = @views step(integ, u, t[it], dt, E[ir,:])
            ne[ir,it+1] = sum(u)

            u_pi = @views step(integ_pi, u_pi, t[it], dt, E[ir,:])
            I = abs2(E[ir,it+1]) * Iu
            Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
            for ic in eachindex(u_pi)
                N0 = N0s[ic]
                R1 = Rs[ic] * tu
                kdne[ir,it+1] += Ks[ic] * R1 * (N0 - u_pi[ic])
            end
        end
    end
    return nothing
end


function solve!(plasma::PlasmaPT4, E::CuArray)
    (; tu, Iu, t, ionrates, N0s, Ks, integs, integs_pi, ne, kdne) = plasma
    @. kdne = 0
    integs = CuArray([integ for integ in integs])
    integs_pi = CuArray([integ for integ in integs_pi])
    Nr = size(E,1)
    @krun Nr solve_pt4_kernel!(ne, kdne, t, E, tu, Iu, ionrates, N0s, Ks, integs, integs_pi)
end
function solve_pt4_kernel!(ne, kdne, t, E, tu, Iu, ionrates, N0s, Ks, integs, integs_pi)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    Nr, Nt = size(E)
    dt = t[2] - t[1]
    for ir=id:stride:Nr
        integ = integs[ir]
        integ_pi = integs_pi[ir]
        u = integ.prob.u0
        u_pi = integ_pi.prob.u0
        ne[ir,1] = sum(u)
        for it=1:Nt-1
            u = @views step(integ, u, t[it], dt, E[ir,:])
            ne[ir,it+1] = sum(u)

            u_pi = @views step(integ_pi, u_pi, t[it], dt, E[ir,:])
            I = abs2(E[ir,it+1]) * Iu
            Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
            for ic in eachindex(u_pi)
                N0 = N0s[ic]
                R1 = Rs[ic] * tu
                kdne[ir,it+1] += Ks[ic] * R1 * (N0 - u_pi[ic])
            end
        end
    end
    return nothing
end
