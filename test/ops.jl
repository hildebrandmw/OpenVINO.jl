@testset "Testing Ops" begin
    @testset "Add" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        # Compile and run
        f = OpenVINO.compile(+, x, y)
        @test isapprox(f(), x + y)

        # Broadcasting variants
        f = OpenVINO.compile((a,b) -> a .+ b, x, y)
        @test isapprox(f(), x + y)
    end

    # Test that constants get slurped up by OpenVINO.
    # TODO: Run over all element types exported by OpenVINO.
    @testset "Constant" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        f = OpenVINO.compile(a -> a .+ y, x)
        @test length(f.request.inputs) == 1
        @test isapprox(f(), x .+ y)

        # Test for broadcasting
        function testfunc1(x)
            return x .+ 1
        end
        f = OpenVINO.compile(testfunc1, x)
        @test isapprox(f(), testfunc1(x))
    end

    @testset "Broadcast" begin
        # Broadcasting for tracing
        x = randn(Float32, 10)
        y = randn(Float32, 10, 10)
        function testfunc1(x, y)
            return x .+ y
        end
        f = OpenVINO.compile(testfunc1, x, y)
        @test isapprox(f(), testfunc1(x, y))
    end

    @testset "Convert Eltype" begin
        x = rand(Int32, 10)
        f = OpenVINO.compile(a -> OpenVINO.convert_eltype(Int64, a), x)
        @test eltype(f()) == Int64
        @test f() == x
    end

    @testset "Divide" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        function testfunc1(a, b)
            return a ./ b
        end
        f = OpenVINO.compile(testfunc1, x, y)
        @test isapprox(f(), testfunc1(x, y))
    end

    @testset "MatMul" begin
        sizes = [
            (10, 1) => (1, 10),
            (10, 10) => (10, 10),
            (10, 10) => (10, 1),
            (1, 10) => (10, 10),
        ]
        for (xsize, ysize) in sizes
            x = randn(Float32, xsize)
            y = randn(Float32, ysize)
            f = OpenVINO.compile(*, x, y)
            @test isapprox(f(), x * y)
        end
    end

    @testset "Log" begin
        # Add 1 to make sure the log is more or less nicely behaved.
        x = rand(Float32, 10, 10) .+ 1
        function testfunc1(a)
            return log.(a)
        end
        f = OpenVINO.compile(testfunc1, x)
        @test isapprox(f(), testfunc1(x))
    end

    @testset "Negative" begin
        # Add 1 to make sure the log is more or less nicely behaved.
        x = randn(Float32, 10, 10)
        f = OpenVINO.compile(-, x)
        @test isapprox(f(), -x)

        function testfunc1(x)
            return (-).(x)
        end
        f = OpenVINO.compile(testfunc1, x)
        @test isapprox(f(), testfunc1(x))
    end

    @testset "Maximum" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        function testfunc1(a, b)
            return max.(a, b)
        end
        f = OpenVINO.compile(testfunc1, x, y)
        @test f() == testfunc1(x, y)
    end

    @testset "Minimum" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        function testfunc1(a, b)
            return min.(a, b)
        end
        f = OpenVINO.compile(testfunc1, x, y)
        @test isapprox(f(), testfunc1(x, y))
    end

    @testset "Multiply" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)

        # Standard elementwise multiply
        function testfunc1(a, b)
            return a .* b
        end
        f = OpenVINO.compile(testfunc1, x, y)
        @test isapprox(f(), testfunc1(x, y))

        # Test broadcasting in both directions
        function testfunc2(a)
            return a .* 2
        end
        f = OpenVINO.compile(testfunc2, x)
        @test isapprox(f(), testfunc2(x))

        function testfunc3(a)
            return 2 .* a
        end
        f = OpenVINO.compile(testfunc3, x)
        @test isapprox(f(), testfunc3(x))
    end

    @testset "Reshape" begin
        tests = Any[
            (100) => (1, :),
            (2, 2, 2) => (:,),
            (3, 2, 1) => (1, 2, 3),
            (1, 2, 3, 4, 5, 6) => (6, 5, :, 3, 2),
        ]

        for (oldshape, newshape) in tests
            x = rand(Float32, oldshape)
            function testfunc1(a)
                return reshape(a, newshape...)
            end
            f = OpenVINO.compile(testfunc1, x)
            @test f() == testfunc1(x)
        end
    end

    # @testset "Sigmoid" begin
    #     x = randn(Float32, 100, 100)
    #     X = nGraph.Node(x)
    #     Y = Flux.σ.(X)
    #     ex = nGraph.compile(backend, [X], [Y])
    #     tX, tY = @tensors backend (x, Y)
    #     ex([tX], [tY])
    #     @test parent(tY) ≈ Flux.σ.(x)

    #     f = nGraph.compile(backend, i -> Flux.σ.(i), x)
    #     @test parent(f()) ≈ Flux.σ.(x)
    # end

    # @testset "Softmax" begin
    #     # 1D case
    #     x = rand(Float32, 100)
    #     z = softmax(x)
    #     f = nGraph.compile(backend, softmax, x)
    #     @test isapprox(z, parent(f()))

    #     # 2D case
    #     x = rand(Float32, 100, 100)
    #     z = softmax(x)
    #     f = nGraph.compile(backend, softmax, x)
    #     @test isapprox(z, parent(f()))
    # end

    @testset "Sqrt" begin
        x = rand(Float32, 10, 10) .+ one(Float32)
        function testfunc1(a)
            return sqrt.(a)
        end
        f = OpenVINO.compile(testfunc1, x)
        @test isapprox(f(), testfunc1(x))
    end

    @testset "Subtract" begin
        x = randn(Float32, 10, 10)
        y = randn(Float32, 10, 10)
        function testfunc1(a, b)
            return a - b
        end
        f = OpenVINO.compile(testfunc1, x, y)
        @test isapprox(f(), testfunc1(x, y))

        # Broadcasting version
        function testfunc2(a, b)
            return a .- b
        end
        f = OpenVINO.compile(testfunc2, x, y)
        @test isapprox(f(), testfunc2(x, y))
    end

    @testset "Sum" begin
        x = randn(Float32, 10, 10)
        functions = [
            sum,
            x -> sum(x; dims = 1),
            x -> sum(x; dims = 2),
            x -> sum(x; dims = (1, 2)),
            x -> sum(x; dims = (2, 1)),
        ]

        for (i, fn) in enumerate(functions)
            @show i
            f = OpenVINO.compile(fn, x)
            @test isapprox(f(), fn(x))
        end
    end
end
