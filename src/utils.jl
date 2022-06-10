# Conversion Pipeline
openvino_convert(x) = x

# Expected Transformation
#
# f(a,b,c) -> Lib.f(ngraph_convert(a), ngraph_convert(b), ngraph_convert(c))
#
# Pretty straightforward.
#
# If it becomes relevant, I've left code in that will do the transformation
#
# f(a,b,c) ->
#    #genysym1 = ngraph_convert(a)
#    #genysym2 = ngraph_convert(b)
#    #genysym2 = ngraph_convert(c)
#    GC.@preserve #gensym1 #gensym2 @gensym3 begin
#       Lib.f(#gensym1, #gensym2, @gensym3)
#    end
macro libcall(expr)
    if expr.head != :call
        error("Only call `@libcall` on function calls")
    end

    # Prefix "Lib." in front of the function call
    fname = expr.args[1]
    if !isa(fname, Symbol)
        error("Don't put the modlue `Lib` in front of the function call")
    end
    fname = :(Lib.$fname)

    args = expr.args[2:end]
    for i in eachindex(args)
        args[i] = :(openvino_convert($(esc(args[i]))))
    end

    return :($fname($(args...)))

    # vars = [gensym() for i in 1:length(args)]

    # # Wrap the arguments in ngraph_convert
    # newargs = []
    # for (var, arg) in zip(vars, args)
    #     newarg = :($(esc(var)) = ngraph_convert($(esc(arg))))
    #     push!(newargs, newarg)
    # end

    # newcall = :($fname($(vars...)))

    # return quote
    #     $(newargs...)
    #     GC.@preserve $(vars...) $newcall
    # end
end
