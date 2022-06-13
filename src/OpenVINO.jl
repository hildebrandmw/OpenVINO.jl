__precompile__(false)
module OpenVINO

import Cassette
import Requires: @require

include("lib.jl")
include("utils.jl")
include("types.jl")
include("ops.jl")

include("compiler/compiler.jl")

# ML Package Compatibility
include("requires.jl")

const GlobalCore = Ref{Core}()

function __init__()
    GlobalCore[] = Core()
end
globalcore() = GlobalCore[]

end # module
