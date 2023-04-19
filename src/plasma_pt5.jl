struct PT5 <: PType end


struct PlasmaPT5{T, G, TF, TA, TU, TN}
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
    Ravas :: TA
    Ks :: TA
    u :: TU
    upi :: TU
    ne :: TN
    kdne :: TN
end

@adapt_structure PlasmaPT5


function Plasma(
    grid::GridRT, field, medium, type::PT5; components, neu=1, mr=1, nuc=0,
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
    ionrates = Tuple(ionrates)   # to avoid allocations and prevent Arrays adaptor

    u = zeros(Nr, Ncomp)
    upi = zeros(Nr, Ncomp)
    ne = zeros(Nr, Nt)
    kdne = zeros(Nr, Nt)

    tu, Iu, neu, dt, mr, nuc = promote(tu, Iu, neu, dt, mr, nuc)

    return PlasmaPT5(
        Ncomp, Nt, tu, Iu, neu, t, dt, mr, nuc, ionrates, N0s, Ravas, Ks, u,
        upi, ne, kdne,
    )
end


function solve!(plasma::PlasmaPT5, E)
    (; Ncomp, tu, Iu, dt, ionrates, N0s, Ravas, Ks, u, upi, ne, kdne) = plasma
    @. ne = 0
    @. kdne = 0
    Nr, Nt = size(E)
    for ir=1:Nr
        for ic=1:Ncomp
            u[ir,ic] = 0
            upi[ir,ic] = 0
            ne[ir,1] += u[ir,ic]
        end
        for it=2:Nt
            I = (abs2(E[ir,it]) + abs2(E[ir,it-1])) / 2 * Iu
            E2 = (real(E[ir,it])^2 + real(E[ir,it-1])^2) / 2
            Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
            for ic=1:Ncomp
                N0 = N0s[ic]
                R1 = Rs[ic] * tu
                R2 = Ravas[ic] * E2
                a = R2 - R1
                if a != 0
                    b = R1 * N0 / a
                    u[ir,ic] = (u[ir,ic] + b) * exp(a * dt) - b
                end
                ne[ir,it] += u[ir,ic]

                upi[ir,ic] = N0 - (N0 - upi[ir,ic]) * exp(-R1 * dt)
                kdne[ir,it] += Ks[ic] * R1 * (N0 - u[ir,ic])
            end
        end
    end
    return nothing
end


function solve!(plasma::PlasmaPT5, E::CuArray)
    Nr = size(E,1)
    @. plasma.ne = 0
    @. plasma.kdne = 0
    @krun Nr solve_pt5_kernel!(plasma, E)
end
function solve_pt5_kernel!(plasma, E)
    id = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    (; Ncomp, tu, Iu, dt, ionrates, N0s, Ravas, Ks, u, upi, ne, kdne) = plasma
    Nr, Nt = size(E)
    for ir=id:stride:Nr
        for ic=1:Ncomp
            u[ir,ic] = 0
            upi[ir,ic] = 0
            ne[ir,1] += u[ir,ic]
        end
        for it=2:Nt
            I = (abs2(E[ir,it]) + abs2(E[ir,it-1])) / 2 * Iu
            E2 = (real(E[ir,it])^2 + real(E[ir,it-1])^2) / 2
            Rs = apply_funcs(ionrates, I)   # see docstring of apply_funcs
            for ic=1:Ncomp
                N0 = N0s[ic]
                R1 = Rs[ic] * tu
                R2 = Ravas[ic] * E2
                a = R2 - R1
                if a != 0
                    b = R1 * N0 / a
                    u[ir,ic] = (u[ir,ic] + b) * exp(a * dt) - b
                end
                ne[ir,it] += u[ir,ic]

                upi[ir,ic] = N0 - (N0 - upi[ir,ic]) * exp(-R1 * dt)
                kdne[ir,it] += Ks[ic] * R1 * (N0 - u[ir,ic])
            end
        end
    end
    return nothing
end
