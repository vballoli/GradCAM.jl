using Flux, Metalhead, Images, Statistics
using Flux: Chain, Dense, Conv
using Metalhead: VGG

function get_backend(arch::String="vgg")
    if arch == "vgg"
        model = VGG19()
    else
        print("Other backends not supported yet")
    end
    return model
end

H, W = 224, 224

function preprocess_input(x::AbstractArray)
    x ./= 127.5
    x .-= 1.
    return x
end

function load_image(path, preprocess::Bool=true)
    image = Images.load(path)
    if preprocess:
        image = reshape(image, (1, size(image)...))
        image = preprocess_input(image)
    return image
end

function deprocess_image(image::AbstractArray)
    if ndims(image) > 3
        image = collect(Iterators.flatten(image))
    end
    x .-= mean(x)
    x ./= (std(x) .+ 1e-5)
    x .*= 0.1

    x .+= 0.5
    x = clamp(x, 0, 1)

    x .*= 255
    x = Int8.(clip(x, 0, 255))
    return x
end

function normalize(x::AbstractArray)
    return (x + 1e-1-) / (sqrt(mean(x^2 + 1e-10)))
end