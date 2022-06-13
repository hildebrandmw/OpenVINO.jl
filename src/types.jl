#####
##### Core
#####

struct Core
    core::Lib.CoreAllocated
end
Core() = Core(Lib.Core())
openvino_convert(core::Core) = core.core

#####
##### Element Type Maps
#####

# This should be kept inline with `openvino/core/type/element_type.hpp`
@enum ElementEnum::Int32 begin
    ov_undefined = 0
    ov_dynamic
    ov_boolean
    ov_bf16
    ov_f16
    ov_f32
    ov_f64
    ov_i4
    ov_i8
    ov_i16
    ov_i32
    ov_i64
    ov_u1
    ov_u4
    ov_u8
    ov_u16
    ov_u32
    ov_u64
end

# This is kind of a gross way of mapping Julia types to ngraph types.
# TODO: Think of a better way of doing this.
const TYPEMAPS = Dict(
    ov_boolean => Bool,
    ov_f32 => Float32,
    ov_f64 => Float64,
    ov_i8 => Int8,
    ov_i16 => Int16,
    ov_i32 => Int32,
    ov_i64 => Int64,
    ov_u8 => UInt8,
    ov_u16 => UInt16,
    ov_u32 => UInt32,
    ov_u64 => UInt64,
)

const Element = Lib.ElementAllocated

# Mapping from Julia => nGraph
Element(::Type{T}) where {T} = error("No translation defined for $T")
for (S, T) in TYPEMAPS
    @eval Element(::Type{$T}) = @libcall openvino_type($(Int32(S)))
end
back(x) = TYPEMAPS[ElementEnum(@libcall type_enum(x))]
openvino_convert(::Type{T}) where {T<:Number} = Element(T)

#####
##### Shape
#####

fillto(len::Integer, value::Integer) = ntuple(_ -> value, Val(len))
function fillto(len::Integer, value::Union{Tuple,AbstractArray})
    repititions, remainder = divrem(len, length(value))
    @assert iszero(remainder)
    iter = Iterators.flatten(Iterators.repeated(value, repititions))
    return (collect(iter)...,)
end

# Automatically convert between row-major and column-major representations
struct Shape{T<:Tuple}
    shape::T

    Shape(shape::NTuple{N,Int}) where {N} = new{NTuple{N,Int}}(shape)
    Shape(shape::T) where {T<:Tuple} = Shape(convert.(Int, shape))
end
Shape(x::AbstractArray) = Shape(size(x))
Shape(shape::Shape) = shape
Shape(len::Integer, x::Union{Integer,Tuple,AbstractArray}) = Shape(fillto(len, x))

openvino_convert(shape::Shape) = collect(reverse(shape.shape))
openvino_convert(::Shape{Tuple{}}) = Vector{Int}()
Base.length(::Shape{T}) where {T} = length(T)

# # Here, is the machinery that does the dispatch.
# _expand(N, x::Integer) = fill(Int(x), N)
#
# # Numbers
# shape(N, x::Number) = _expand(N, x)
# strides(N, x::Number) = _expand(N, x)
#
# # Tuples and Vectors
# maybecollect(x::Vector) = x
# maybecollect(x) = collect(x)
#
# function ordered(N, x)
#     # Determine how many repetitions we need
#     repitions, remainder = divrem(length(x), N)
#     if !iszero(remainder)
#         error("The length of `x` must be divisible by $N")
#     end
#
#     return _reversed(repeat(maybecollect(x), repitions))
# end
# shape(N, x::Union{Tuple,Vector}) = ordered(N, x)
# strides(N, x::Union{Tuple,Vector}) = ordered(N, x)

#####
##### Strides
#####

# Automatically convert between row-major and column-major representations
struct Strides{T<:Tuple}
    strides::T

    Strides(strides::NTuple{N,Int}) where {N} = new{NTuple{N,Int}}(strides)
    Strides(strides::T) where {T<:Tuple} = Strides(convert.(Int, strides))
end
Strides(x::AbstractArray{T}) where {T} = Strides(sizeof(T) .* strides(x))
Strides(strides::Strides) = strides
Strides(len::Integer, x::Union{Integer,Tuple,AbstractArray}) = Strides(fillto(len, x))
function openvino_convert(strides::Strides{T}) where {T}
    return collect(reverse(strides.strides))
end
openvino_convert(::Strides{Tuple{}}) = Int[]

#####
##### Node
#####

const NodeCpp = Lib.CxxWrap.StdLib.SharedPtrAllocated{Lib.Node}
Base.ndims(x::NodeCpp) = length(@libcall get_output_shape(x, 0))
Base.eltype(x::NodeCpp) = back(@libcall get_output_element_type(x, 0))

struct Node{T,N} <: AbstractArray{T,N}
    node::NodeCpp
end
openvino_convert(node::Node) = node.node

# For some reason, CxxWrap interprets the Julia side of
# "jlcxx::ArrayRef<std::shared_ptr<ov::Node>>" as a
# "Vector{CxxWrap.CxxWrap{StdLib.SharedPtr{Node}}}" instead of accepting
# "Vector{StdLib.SharedPtr{NodeAllocated}".
#
# Consequently, we need to wrap all the nodes in a "CxxRef" in order to make CxxWrap happy.
openvino_convert(v::AbstractVector{Node}) = Lib.CxxWrap.CxxRef.(openvino_convert.(v))

Base.show(io::IO, x::Node{T,N}) where {T,N} = print(io, "Node{$T,$N} - $(name(x))")
Base.display(x::Node) = show(stdout, x)

Node(x::Node) = x
Node{T}(x::NodeCpp) where {T} = Node{T,ndims(x)}(x)
Node(x::NodeCpp) = Node{eltype(x),ndims(x)}(x)

Node(x::AbstractArray{T,N}) where {T,N} = Node{T,N}(x)
Node{T,N}(x::AbstractArray{T,N}) where {T,N} = parameter(T, size(x))

Node(x::T) where {T<:Number} = Node{T}(x)
Node{T}(x::T) where {T<:Number} = Node{T,0}(fill(x))

# Array style arguments
Base.ndims(x::Node) = ndims(x.node)

function Base.size(x::Node{T,N}) where {T,N}
    @assert N == ndims(x)
    dims = @libcall get_output_shape(x, 0)
    return ntuple(i -> convert(Int, dims[N + 1 - i]), Val(N))
end

Base.size(x::Node, i::Integer) = size(x)[i]
Base.length(x) = prod(size(x))
Base.eltype(x::Node{T}) where {T} = T
Base.IndexStyle(::Node) = Base.IndexLinear()

name(x::Node) = String(@libcall get_name(x))
description(x::Node) = String(@libcall description(x))

# So these can be used as keys in a Dict
Base.:(==)(x::T, y::T) where {T<:Node} = name(x) == name(y)
Base.hash(x::Node, h::UInt = 0) = hash(name(x), h)

function Base.getindex(::Node, i::Int)
    return error(
        "Yeah, yeah, I know \"Node\" is an AbstractArray ... but please don't index into it.",
    )
end

#####
##### Tensors
#####

struct Tensor{T,N} <: DenseArray{T,N}
    data::Array{T,N}
    handle::Lib.Tensor
end
openvino_convert(tensor::Tensor) = tensor.handle

Tensor(tensor::Tensor) = tensor
function Tensor(data::Array{T,N}) where {T,N}
    handle = GC.@preserve data @libcall create_tensor(
        T, Shape(data), convert(Ptr{Nothing}, pointer(data)), Strides(data)
    )
    return Tensor(data, handle)
end

# Construct Tensors from Nodes
function Tensor(node::Node{T,N}) where {T,N}
    array = Array{T,N}(undef, size(node))
    return Tensor(array)
end

# Implement the `AbstractArray` interface for tensors.
Base.size(tensor::Tensor) = size(tensor.data)
Base.getindex(tensor::Tensor, i::Int) = tensor.data[i]
Base.setindex!(tensor::Tensor, v, i::Int) = (tensor.data[i] = v)
Base.IndexStyle(::Type{<:Tensor}) = Base.IndexLinear()
Base.similar(tensor::Tensor) = Tensor(similar(tensor.data))

#####
##### Model
#####

struct Model
    model::Lib.CxxWrap.StdLib.SharedPtrAllocated{Lib.Model}
end
openvino_convert(model::Model) = model.model

# Read model from file.
Model(path::String; core = globalcore()) = Model(@libcall read_model(core, path))
function Model(inputs::AbstractVector{Node}, outputs::AbstractVector{Node})
    model = GC.@preserve inputs outputs begin
        @libcall make_model(outputs, inputs)
    end
    return Model(model)
end

function compile(model::Model; core = globalcore(), device = "CPU")
    return CompiledModel(core, model, device)
end
print_io(model::Model) = @libcall print_io(model)

#####
##### CompiledModel
#####

struct CompiledModel
    model::Lib.CompiledModelAllocated
end

openvino_convert(model::CompiledModel) = model.model
function CompiledModel(core::Core, model::Model, device = "CPU")
    return CompiledModel(@libcall compile(core, model, device))
end

#####
##### InferRequest
#####

struct InferRequest
    request::Lib.InferRequestAllocated
    inputs::Vector{Any}
    outputs::Vector{Any}
end
openvino_convert(request::InferRequest) = request.request
function InferRequest(model::CompiledModel)
    return InferRequest(@libcall(create_infer_request(model)), Any[], Any[])
end

function setinput!(request::InferRequest, tensor::Tensor, idx = 1)
    @libcall set_input_tensor(request, idx - one(idx), tensor)
    (; inputs) = request
    if idx > length(inputs)
        resize!(inputs, idx)
    end
    inputs[idx] = tensor
    return nothing
end
function setoutput!(request::InferRequest, tensor::Tensor, idx = 1)
    @libcall set_output_tensor(request, idx - one(idx), tensor)
    (; outputs) = request
    if idx > length(outputs)
        resize!(outputs, idx)
    end
    outputs[idx] = tensor
    return nothing
end

(request::InferRequest)() = @libcall infer(request)

