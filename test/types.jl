@testset "Testing Elements" begin
    elements = [
        Bool,
        #Float16,
        Float32,
        Float64,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
    ]

    # Does this survive the round trip?
    for element in elements
        @test OpenVINO.back(OpenVINO.Element(element)) == element
    end
end

@testset "Testing Nodes" begin
    # Create a dummy parameter.
    param = OpenVINO.parameter(Float32, (10, 10))

    @test isa(param, OpenVINO.Node)

    # Test printing
    println(devnull, param)

    @test ndims(param) == 2
    @test size(param) == (10,10)
    @test size(param, 1) == 10
    @test size(param, 2) == 10
    @test eltype(param) == Float32
    @test eltype(param.node) == Float32
    @test OpenVINO.description(param) == "Parameter"
    println(devnull, OpenVINO.name(param))

    # Go into higher dimensions.
    x = OpenVINO.parameter(Int32, (10, 20, 30))
    @test size(x) == (10, 20, 30)
    @test size(x, 3) == 30
end
