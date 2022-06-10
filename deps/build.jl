using CxxWrap

@info "Building Lib"

# Paths for linking
cxxhome = CxxWrap.prefix_path()
juliahome = dirname(Base.Sys.BINDIR)
openvinohome = joinpath(@__DIR__, "l_openvino_toolkit_runtime_ubuntu20_p_2022.1.0.643", "runtime")

# Use Clang since it seems to get along better with Julia
cxx = "clang++"

cxxflags = [
    "-g",
    "-O3",
    "-Wall",
    "-fPIC",
    "-std=c++17",
    "-DPCM_SILENT",
    "-DJULIA_ENABLE_THREADING",
    "-Dexcept_EXPORTS",
    # Surpress some warnings from Cxx
    "-Wno-unused-variable",
    "-Wno-unused-lambda-capture",
]

includes = [
    "-I$(joinpath(cxxhome, "include"))",
    "-I$(joinpath(juliahome, "include", "julia"))",
    "-I$(joinpath(openvinohome, "include"))",
    "-I$(joinpath(openvinohome, "include", "ie"))",
]

loadflags = [
    # Linking flags for Julia
    "-L$(joinpath(juliahome, "lib"))",
    "-Wl,--export-dynamic",
    "-Wl,-rpath,$(joinpath(juliahome, "lib"))",
    "-ljulia",
    # Linking Flags for CxxWrap
    "-L$(joinpath(cxxhome, "lib"))",
    "-Wl,-rpath,$(joinpath(cxxhome, "lib"))",
    "-lcxxwrap_julia",
    # Linking Flags for nGraph
    "-L$(joinpath(openvinohome, "lib", "intel64"))",
    "-Wl,-rpath,$(joinpath(openvinohome, "lib", "intel64"))",
    "-lopenvino",
    # Add TBB
    "-L$(joinpath(openvinohome, "3rdparty", "tbb", "lib"))",
    "-Wl,-rpath,$(joinpath(openvinohome, "3rdparty", "tbb", "lib"))",
    "-ltbb",
]

src = joinpath(@__DIR__, "openvino-julia.cpp")
so = joinpath(@__DIR__, "libopenvino-julia.so")

cmd = `$cxx $cxxflags $includes -shared $src -lpthread -o $so $loadflags`
@show cmd
run(cmd)

