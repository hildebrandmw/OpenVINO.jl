# Some tuple utilities
astuple(x::Tuple) = x
astuple(x) = (x,)

untuple(x::Tuple) = x
untuple(x::Tuple{T}) where {T} = first(x)

unwrap_zerodim(x::AbstractArray) = x
unwrap_zerodim(x::AbstractArray{T,0}) where {T} = x[]

# NOTE: Register was required in the past for some nodes like `BatchNorm` that left
# dangling nodes in the computation graph that then led ngraph to segfault
#
# TODO: Does this still happen?
"""
    __register(x::Node) -> Nothing

By default, becomes a no-op. When executed under [`CompileCtx`](@ref), registers `x` as a
hidden output of a ngraph function.
"""
__register(::Node) = nothing

#####
##### Cassette Magic
#####

Cassette.@context CompileCtx
struct CompileMetadata
    training::Bool

    # Mapping of a Parameter array to nGraph Node
    array_to_node::IdDict{Any,Node}

    # Keep track of the constants we have created.
    constants::IdDict{Any,Node}
end

function CompileMetadata(array_to_node, training::Bool)
    return CompileMetadata(training, array_to_node, IdDict{Any,Node}())
end

# Flag to indicate if we are training or not.
#
# If so, we need to add another pass to compute the gradients of all parameters.
istraining(x::CompileCtx) = x.metadata.training

#####
##### Cassette Overdubs
#####

function Base.get(ctx::CompileCtx, x::AbstractArray) where {T,N}
    # Check if this array is a parameter.
    # If so, get our cached input for it.
    node = get(ctx.metadata.array_to_node, x, nothing)
    if !isnothing(node)
        return node
    end

    # Check if we've already made a constant for this object.
    node = get(ctx.metadata.constants, x, nothing)
    if isnothing(node)
        node = constant(x)
        ctx.metadata.constants[x] = node
    end
    return node
end

function Base.get(::Type{Node{T,N}}, ctx::CompileCtx, x::AbstractArray) where {T,N}
    node = get(ctx, x)
    @assert isa(node, Node{T,N})
    return node
end

# Hijack Node constructors from Arrays
function Cassette.overdub(ctx::CompileCtx, ::Type{Node{T,N}}, x::AbstractArray) where {T,N}
    return get(Node{T,N}, ctx, x)
end

# Do not hijack creating nodes from nodes.
Cassette.overdub(ctx::CompileCtx, f::Type{<:Node}, x::Node) = f(x)

# Get around Cassette bug with `reverse`
Cassette.overdub(::CompileCtx, ::typeof(reverse), args...) = reverse(args...)

# # Hijack these layers
# Cassette.overdub(ctx::CompileCtx, f::Flux.Dense, args...) =
#     Cassette.overdub(ctx, _dense_impl, f, args...)

# function Cassette.overdub(
#     ctx::CompileCtx, f::T, x
# ) where {T<:Union{Flux.Conv,Flux.CrossCor}}
#     return convolution_implementation(
#         x,
#         get(ctx, f.weight),
#         get(ctx, f.bias);
#         f.stride,
#         f.pad,
#         f.dilation,
#         activation = f.σ,
#     )
# end

# Cassette.overdub(ctx::CompileCtx, f::Flux.CrossCor, args...) =
#     Cassette.overdub(ctx, _conv_impl, f, args...)
#
# Cassette.overdub(ctx::CompileCtx, f::Flux.BatchNorm, args...) =
#     Cassette.overdub(ctx, _batchnorm_impl, f, args...)

# Short circuits to reduce compile times
Cassette.overdub(::CompileCtx, f::typeof(rand), args...) = f(args...)
#Cassette.overdub(ctx::CompileCtx, f::typeof(Flux.glorot_normal), args...) = f(args...)
#Cassette.overdub(ctx::CompileCtx, f::typeof(Flux.glorot_uniform), args...) = f(args...)

#####
##### Main `compile` entrypoint
#####

"""
    compile(backend, f, x...; [training]) -> Executable

Trace and compile a Flux model `f` with `args`.
"""
function compile(f, x...; kw...)
    # Build the nGraph computation graph
    trace, _ = snoop(f, x...; kw...)

    # pass the trace as well as the optimizer arguments to create the executable.
    return make(trace; kw...)
end

"""
Traced parameters from a compiled function: `f(args...)`.

* `inputs`: The nodes that are explicitly used as an input to `f`. Corresponds to
    `args...` in the signature of `f`.
* `outputs`: The nodes that represent the results of `f(args...)`.
"""
struct Trace
    input_arrays::Vector{Any}
    input_nodes::Vector{Node}
    output_nodes::Vector{Node}

    # Record of what is a parameter
    parameter_arrays::Vector{Any}
    parameter_nodes::Vector{Node}

    array_to_node::IdDict{Any,Node}
    node_to_array::IdDict{Node,Any}
end

# Step 1 of compilation - trace the provided function with the provided arguments to
# construct an nGraph graph - apply the requested optimizer to finish graph construction.
#
# Returns an argument named tuple that is given to the next stage of compilation.
# snoop(f, x...; kw...) = snoop(f, Flux.Params(), x...; kw...)
function snoop(f, xs...; training = false, parameters = (), kw...)
    # Extract the inputs and other parameters
    input_arrays = Any[x for x in xs]
    input_nodes = [Node(x) for x in xs]
    parameter_nodes = isempty(parameters) ? Node[] : [Node(p) for p in parameters]

    array_to_node = IdDict{Any,Node}()
    node_to_array = IdDict{Node,Any}()

    # Construct parameter-to-node mappings.
    for (node, array) in zip(parameter_nodes, parameters)
        array_to_node[array] = node
        node_to_array[node] = array
    end

    metadata = CompileMetadata(array_to_node, training)
    ctx = CompileCtx(; metadata)

    # Perform traced execution on the function.
    output_nodes = collect(astuple(Cassette.overdub(ctx, f, input_nodes...)))
    @assert all(x -> isa(x, Node), output_nodes)

    trace = Trace(
        input_arrays,
        input_nodes,
        output_nodes,
        collect(parameters),
        parameter_nodes,
        array_to_node,
        node_to_array,
    )

    return (trace, metadata)
end

function make(trace::Trace; training = false, kw...)
    (;
        input_arrays,
        input_nodes,
        output_nodes,
        parameter_arrays,
        parameter_nodes,
    ) = trace
    # Sanity check on inputs
    @assert length(parameter_arrays) == length(parameter_nodes)
    for (_array, _node) in zip(parameter_arrays, parameter_nodes)
        @assert size(_array) == size(_node)
    end

    # Create an nGraph Executable
    allinputs = [input_nodes; parameter_nodes]
    model = Model(allinputs, output_nodes)
    compiled_model = compile(model)
    request = InferRequest(compiled_model)

    for (i, input) in enumerate(input_arrays)
        setinput!(request, Tensor(input), i)
    end

    base = length(input_arrays)
    for (i, parameter) in enumerate(parameter_arrays)
        setinput!(request, Tensor(parameter), i + base)
    end

    for (i, output) in enumerate(output_nodes)
        setoutput!(request, Tensor(output), i)
    end

    return CompiledFunction(request)
end

#####
##### OpenVINOFunction
#####

struct CompiledFunction{T <: Tuple}
    request::InferRequest
    outputs::T
end

CompiledFunction(request::InferRequest) = CompiledFunction(request, (request.outputs...,))

function (fn::CompiledFunction)()
    (; request, outputs) = fn
    request()
    return untuple(unwrap_zerodim.(outputs))
end

# #####
# ##### Executable
# #####
#
# struct CallableFunction{O}
#     ex::Executable
#     inputs::Vector{Tensor}
#     outputs::NTuple{O,Tensor}
#     implicit_outputs::Vector{Tensor}
# end
#
# allinputs(x::CallableFunction) = x.inputs
# function alloutputs(x::CallableFunction)
#     return collect(Iterators.flatten((x.outputs, x.implicit_outputs)))
# end
#
# function (ex::CallableFunction)()
#     inputs = allinputs(ex)
#     outputs = alloutputs(ex)
#
#     # Since we're passing wrapped type to C++, we have to cast them to Any's which is
#     # kind of gross - but w/e
#     ex.ex(inputs, outputs)
#     return untuple(ex.outputs)
# end
#
