using Flux, Zygote
using Flux: Chain, Dense, Conv, onehotbatch
using Zygote: @adjoint

@adjoint relu(x) = relu(x), Δ -> ((Δ .* (Δ .> 0)) * (x .* (x .> 0)) .* Δ, -1)

function guided_backprop()
    
end
