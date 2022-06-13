# Flux compatibility
@require NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd" begin
    # Activation Functions
    NNlib.relu(x::Node) = relu(x)
    NNlib.σ(x::Node) = sigmoid(x)

    # Layers
    function NNlib.conv(
        x::Node{xT,N}, weight::Node{wT,N}, dims::NNlib.ConvDims
    ) where {xT,wT,N}
        return convolution(
            x,
            weight;
            stride = NNlib.stride(dims),
            pad = NNlib.padding(dims),
            dilation = NNlib.dilation(dims),
        )
    end

    function NNlib.maxpool(x::Node{T,N}, dims::NNlib.PoolDims) where {T,N}
        return maxpool(
            x,
            NNlib.kernel_size(dims);
            pad = NNlib.padding(dims),
            stride = NNlib.stride(dims),
            dilation = NNlib.dilation(dims),
        )
    end

    function NNlib.meanpool(x::Node{T,N}, dims::NNlib.PoolDims) where {T,N}
        return avgpool(
            x,
            NNlib.kernel_size(dims);
            pad = NNlib.padding(dims),
            stride = NNlib.stride(dims),
            #dilation = NNlib.dilation(dims),
        )
    end

    # Compiler Plugins
    function Cassette.overdub(
        ctx::CompileCtx, f::typeof(NNlib.conv), _x, _weight, dims::NNlib.ConvDims
    )
        x = get(ctx, _x)
        weight = get(ctx, _weight)
        # The semantics of flipping kernels are reversed between `NNlib` and `OpenVINO`.
        if !NNlib.flipkernel(dims)
            weight = reverse_dims(weight, (1, 2))
        end
        return NNlib.conv(get(ctx, x), get(ctx, weight), dims)
    end
end

@require Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c" begin
    # Compiler Plugins
    function Cassette.overdub(ctx::CompileCtx, BN::Flux.BatchNorm, x::Node)
        (; λ, ϵ) = BN
        β = get(ctx, BN.β)
        γ = get(ctx, BN.γ)
        μ = get(ctx, BN.μ)
        σ² = get(ctx, BN.σ²)
        return λ.(batchnorm_inference(x, γ, β, μ, σ², ϵ))
    end
end
