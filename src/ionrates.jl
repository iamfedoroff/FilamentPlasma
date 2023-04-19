struct IonizationRate{T, A<:AbstractVector{T}} <: Function
    x :: A
    y :: A
end

@adapt_structure IonizationRate


function IonizationRate(fname::String)
    data = readdlm(fname)
    x = data[:,1]
    y = data[:,2]

    @. x = log10(x)
    @. y = log10(y)

    # Check for sorted and evenly spaced x values:
    dx = diff(x)
    @assert issorted(x)
    @assert all(x -> isapprox(x, dx[1]), dx)

    return IonizationRate(x, y)
end


function (tf::IonizationRate{T})(x::T) where T
    if x <= 0
        y = zero(T)   # in order to avoid -Inf in log10(0)
    else
        xlog10 = log10(x)
        ylog10 = linterp(xlog10, tf.x, tf.y)
        y = 10^ylog10
    end
    return y
end
