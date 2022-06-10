__precompile__(false)
module OpenVINO

import Cassette

include("lib.jl")
include("utils.jl")
include("types.jl")
include("ops.jl")

include("compiler/compiler.jl")

const GlobalCore = Ref{Core}()

function __init__()
    GlobalCore[] = Core()
end
globalcore() = GlobalCore[]

end # module
