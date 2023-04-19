struct PT2 <: PType end


struct PlasmaPT2{T, G, TI, TN}
    Nt :: Int
    Ncomp :: Int
    dt :: T
    t :: G
    integs :: TI
    neu :: T
    ne :: TN
end

@adapt_structure PlasmaPT2


function Plasma(
    grid::Grid, field, medium, type::PT2; components, alg=RK3(), neu=nothing,
)
    (; Nt, tu, t, dt) = grid
    (; Iu, E) = field
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

    integs = Array{Integrator}(undef, Ncomp)
    for ic=1:Ncomp
        u0 = 0.0
        prob = Problem(func_pt2, u0, (tu, Iu, N0s[ic], ionrates[ic], t))
        integs[ic] = Integrator(prob, alg)
    end
    integs = Tuple(integs)
    # integs = CuArray(hcat([integs[i] for i in 1:Ncomp]))

    ne = zeros(size(E))

    return PlasmaPT2(Nt, Ncomp, dt, t, integs, neu, ne)
end


# ******************************************************************************
function func_pt2(u, p, t, EE)
    tu, Iu, N0, ionrate, tt = p
    E = linterp(t, tt, EE)
    I = abs2(E) * Iu
    R = ionrate(I) * tu
    du = R * (N0 - u)
    return du
end


function solve!(plasma::PlasmaPT2, E::Matrix)
    (; Ncomp, dt, t, integs, ne) = plasma
    Nr, Nt = size(E)
    @. ne = 0
    for ir=1:Nr
        for ic=1:Ncomp
            integ = integs[ic]
            u = integ.prob.u0
            ne[ir,1] += u
            for it=1:Nt-1
                u = @views step(integ, u, t[it], dt, E[ir,:])
                ne[ir,it+1] += u
            end
        end
    end
    return nothing
end


function solve!(plasma::PlasmaPT2, E::CuArray)
    (; t, integs, ne) = plasma
    @. ne = 0
    Nr = size(E, 1)

    integs = CuArray([integ for integ in integs])

    @krun Nr solve_pt2_kernel!(ne, t, E, integs)
    return nothing
end
function solve_pt2_kernel!(ne, t, E, integs)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    Nr, Nt = size(ne)
    Ncomp = length(integs)
    dt = t[2] - t[1]
    for ir=id:stride:Nr
        for ic=1:Ncomp
            integ = integs[ic]    # DOES NOT WORK!
            # u = integ.prob.u0
            # ne[ir,1] += u
            # for it=1:Nt-1
            #     u = @views step(integ, u, t[it], dt, E[ir,:])
            #     ne[ir,it+1] += u
            # end
        end
    end
    return nothing
end
