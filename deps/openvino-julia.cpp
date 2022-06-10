// stdlib
#include <algorithm>
#include <iostream>
#include <type_traits>
#include <vector>

// jlcxx
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

// openvino
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset8.hpp"

template <typename T> void reverse(T& x) { std::reverse(x.begin(), x.end()); }

// Convert a Julia array of UInt8 to a std::vector of type `T`.
//
// Used for the construction of Constants.
template <typename T> std::vector<T> castcollect(const jlcxx::ArrayRef<uint8_t>& x) {
    // Get the raw data from the array and reinterpret it to T*
    const T* first = reinterpret_cast<const T*>(x.data());
    const T* last = reinterpret_cast<const T*>(x.data() + x.size());
    return std::vector(first, last);
}

template <typename T, typename U>
std::vector<T> convertcollect(const jlcxx::ArrayRef<U>& x) {
    return std::vector<T>(x.begin(), x.end());
}

// Since C++ does not allow partial function template specialization, we use
// this intermediate `type` struct to propagate type information into function
// arguments in order to correctly dispatch between the various ngraph
// `std::vector` derived classes and other constructs like `ngraph::AxisSet`.
template <typename T> struct type {};

// `tovector`
//
// The goal of this suite of funtions is to convert the various OpenVINO types like
// `ov::Shape` and `ov::Strides` and copy them into a standard vector, possibly allowing
// for a change in element type as well.
template <typename U, typename T> std::vector<U> tovector(type<U>, const T& x) {
    return std::vector<U>(x.begin(), x.end());
}

template <typename T> auto tovector(const T& x) {
    return tovector(type<typename T::value_type>(), x);
}

template <> auto tovector(const ov::AxisSet& x) { return x.to_vector(); }

// `construct`
//
// This is basically the opposite of `tovector`.
// Instead of converting OpenVINO types to standard types, this family of functions
// converts standard and Julia types to OpenVINO types
template <typename T, typename U> T construct(type<T>, U x) {
    return T(x.begin(), x.end());
}

template <typename U> ov::AxisSet construct(type<ov::AxisSet>, const U& x) {
    return ov::AxisSet(convertcollect<size_t>(x));
}

template <typename T, typename U> T construct(const U& x) {
    return construct(type<T>(), x);
}

/////
///// ov::Node casting
/////

template <typename> struct is_shared_ptr : std::false_type {};
template <typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {
    using parameter_type = T;
};

template <typename T> struct remove_constref {
    using type = typename std::remove_const<typename std::remove_reference<T>::type>::type;
};

// Remove `const` and `reference` type modifiers.
template <typename T> using remove_constref_t = typename remove_constref<T>::type;

// Return `true` if `T` is a shared pointer.
template <typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<remove_constref_t<T>>::value;

// Obtain the type of an object pointed to by a shared pointer.
template <typename T>
using shared_ptr_t = typename is_shared_ptr<remove_constref_t<T>>::parameter_type;

// Return `true` if `T` is a shared pointer that points to a class inheriting from
// `ov::Node`
template <typename T>
inline constexpr bool is_node_ptr = std::is_base_of<ov::Node, shared_ptr_t<T>>::value;

// Convert a shared pointer of some object inheriting from `ov::Node` to a
// `std::shared_ptr<ov::Node>`. Should work for
// - lvalue references
// - const lvalue references
// - rvalue references
template <typename T> std::shared_ptr<ov::Node> tonode(T&& x) {
    static_assert(is_shared_ptr_v<T>);
    static_assert(is_node_ptr<T>);
    return std::dynamic_pointer_cast<ov::Node>(std::forward<T>(x));
}

/////
///// Module Wrapping
/////

// Forward to node creation and auto-cast to `op::Node`
#define opset opset8
#define makeop(name, ...) tonode(std::make_shared<ov::opset::name>(__VA_ARGS__))

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    /////
    ///// Elements
    /////

    // The basic strategy is to map back and forth using enum values.
    mod.add_type<ov::element::Type>("Element")
        .method("c_type_name", &ov::element::Type::c_type_string)
        .method("type_enum", [](const ov::element::Type& type) {
            ov::element::Type_t type_t{type};
            static_assert(sizeof(ov::element::Type_t) == sizeof(int));
            return static_cast<int>(type_t);
        });

    // Each of the openvino types has a Enum value.
    // Here, we take the integer value from the Julia shadow of the type
    // selector enum, cast it to the C++ enum and use that to return the
    // reference to the correct type.
    mod.method("openvino_type", [](int32_t enum_value) {
        ov::element::Type_t type_t{enum_value};
        return ov::element::Type(type_t);
    });

    /////
    ///// Node
    /////

    mod.add_type<ov::Node>("Node")
        .method("get_output_size", &ov::Node::get_output_size)
        .method(
            "get_output_shape",
            [](const std::shared_ptr<ov::Node>& node, int64_t index) {
                return dynamic_cast<const std::vector<size_t>&>(
                    node->output(index).get_shape()
                );
            }
        )
        .method("get_output_element_type", &ov::Node::get_output_element_type)
        .method("get_name", &ov::Node::get_name)
        .method("description", &ov::Node::description);

    /////
    ///// NodeOutput
    /////

    mod.add_type<ov::Output<ov::Node>>("NodeOutput")
        .constructor<const std::shared_ptr<ov::Node>&, size_t>()
        .method("get_node", &ov::Output<ov::Node>::get_node_shared_ptr)
        .method("get_index", &ov::Output<ov::Node>::get_index)
        .method("get_shape", [](const ov::Output<ov::Node>& output) {
            auto vector = tovector(output.get_shape());
            // Reverse the vector to convert from OpenVINO's row-major form to
            // Julia's column-major form.
            reverse(vector);
            return vector;
        });

    // We occaisionally hand back `std::vectors` of `Node` shared_pointers.
    // Here, we opt into the stl in CxxWrap.
    jlcxx::stl::apply_stl<std::shared_ptr<ov::Node>>(mod);

    /////
    ///// Function
    /////

    mod.add_type<ov::Model>("Model")
        .method("get_name", &ov::Model::get_name)
        .method(
            "num_inputs",
            [](const std::shared_ptr<ov::Model>& model) { return model->inputs().size(); }
        )
        .method(
            "num_outputs",
            [](const std::shared_ptr<ov::Model>& model) { return model->outputs().size(); }
        )
        .method(
            "get_results",
            [](const std::shared_ptr<ov::Model>& fn) {
                // Need to convert `ov::op::Result` to just `ov::Node`
                auto results = fn->get_results();
                auto x = std::vector<std::shared_ptr<ov::Node>>(results.size());
                std::transform(
                    results.begin(),
                    results.end(),
                    x.begin(),
                    [](const std::shared_ptr<ov::opset::Result>& x) { return tonode(x); }
                );
                return x;
            }
        )
        .method("print_io", [](const std::shared_ptr<ov::Model>& model) {
            for (const auto& input : model->inputs()) {
                std::cout << "    inputs" << std::endl;
                const auto name = input.get_names().empty() ? "NONE" : input.get_any_name();
                std::cout << "        input name: " << name << std::endl;
                std::cout << "        input type: " << input.get_element_type()
                          << std::endl;
                std::cout << "        input shape: " << input.get_shape() << std::endl;
            }

            for (const auto& output : model->outputs()) {
                std::cout << "    outputs" << std::endl;
                const auto name =
                    output.get_names().empty() ? "NONE" : output.get_any_name();
                std::cout << "        output name: " << name << std::endl;
                std::cout << "        output type: " << output.get_element_type()
                          << std::endl;
                std::cout << "        output shape: " << output.get_shape() << std::endl;
            }
        });

    mod.method(
        "make_model",
        [](const jlcxx::ArrayRef<std::shared_ptr<ov::Node>> jl_results,
           const jlcxx::ArrayRef<std::shared_ptr<ov::Node>> jl_parameters) {
            // Convert the Julia Arrays of results and parameters to the correct types
            auto results = ov::OutputVector(jl_results.begin(), jl_results.end());

            // For the Parameters, we have to cast the nodes to `Parameters`
            auto op = [](const std::shared_ptr<ov::Node>& x) {
                return std::dynamic_pointer_cast<ov::opset::Parameter>(x);
            };
            auto parameters = ov::ParameterVector(jl_parameters.size());
            std::transform(
                jl_parameters.begin(),
                jl_parameters.end(),
                parameters.begin(),
                op
            );
            return std::make_shared<ov::Model>(std::move(results), std::move(parameters));
        }
    );

    /////
    ///// runtime::Tensor
    /////

    // Define before `Executable`, since these come as arguments to the
    // `Executable`.
    mod.add_type<ov::Tensor>("Tensor")
        .method(
            "get_shape",
            [](const ov::Tensor& tensor) {
                return tovector(type<int64_t>(), tensor.get_shape());
            }
        )
        .method("get_size_in_bytes", &ov::Tensor::get_byte_size)
        .method("get_element_type", &ov::Tensor::get_element_type);

    mod.method(
        "create_tensor",
        [](const ov::element::Type& type,
           const jlcxx::ArrayRef<int64_t> shape,
           void* host_ptr,
           jlcxx::ArrayRef<int64_t> strides) {
            return ov::Tensor(
                type,
                construct<ov::Shape>(shape),
                host_ptr,
                construct<ov::Strides>(strides)
            );
        }
    );

    /////
    ///// CompiledModel
    /////

    mod.add_type<ov::Core>("Core");
    mod.add_type<ov::InferRequest>("InferRequest")
        .method("infer", &ov::InferRequest::infer)
        .method(
            "set_input_tensor",
            [](ov::InferRequest& infer_request, const ov::Tensor& tensor) {
                infer_request.set_input_tensor(tensor);
            }
        )
        .method(
            "set_input_tensor",
            [](ov::InferRequest& infer_request, size_t idx, const ov::Tensor& tensor) {
                infer_request.set_input_tensor(idx, tensor);
            }
        )
        .method(
            "set_output_tensor",
            [](ov::InferRequest& infer_request, const ov::Tensor& tensor) {
                infer_request.set_output_tensor(tensor);
            }
        )
        .method(
            "set_output_tensor",
            [](ov::InferRequest& infer_request, size_t idx, const ov::Tensor& tensor) {
                infer_request.set_output_tensor(idx, tensor);
            }
        );

    mod.method("read_model", [](ov::Core& core, const std::string& path) {
        return core.read_model(path);
    });

    // Needs to be defined before `backend->compile` because `compile` returns
    // an Executable.
    mod.add_type<ov::CompiledModel>("CompiledModel")
        .method("create_infer_request", &ov::CompiledModel::create_infer_request);
    mod.method(
        "compile",
        [](ov::Core& core,
           const std::shared_ptr<ov::Model>& model,
           const std::string& device_name) {
            return core.compile_model(model, device_name);
        }
    );
    // mod.method("create_infer_request", [](ov::CompiledModel& model) {
    //     return model.create_infer_request();
    // });

    // // TODO: We might be able to optimize this by pre-creating the
    // `std::vector`
    // // and just passing those.
    // .method("call", [](const std::shared_ptr<ov::runtime::CompiledModel>
    // executable,
    //             const
    //             jlcxx::ArrayRef<std::shared_ptr<ngraph::runtime::Tensor>>
    //             jl_outputs, const
    //             jlcxx::ArrayRef<std::shared_ptr<ngraph::runtime::Tensor>>
    //             jl_inputs)
    // {
    //     auto inputs = std::vector<std::shared_ptr<ngraph::runtime::Tensor>>(
    //         jl_inputs.begin(),
    //         jl_inputs.end()
    //     );

    //     auto outputs = std::vector<std::shared_ptr<ngraph::runtime::Tensor>>(
    //         jl_outputs.begin(),
    //         jl_outputs.end()
    //     );

    //     executable->call(outputs, inputs);
    // });

    // /////
    // ///// Backend
    // /////

    // mod.add_type<ngraph::runtime::Backend>("Backend")
    //     .method("compile", [](
    //         const std::shared_ptr<ngraph::runtime::Backend>& backend,
    //         const std::shared_ptr<ngraph::Function>& func,
    //         bool enable_performance_data)
    //     {
    //         return backend->compile(func, enable_performance_data);
    //     })
    //     .method("remove_compiled_function",
    //     &ngraph::runtime::Backend::remove_compiled_function)
    //     .method("get_version", &ngraph::runtime::Backend::get_version);

    // mod.method("create", [](const std::string& type){
    //     return ngraph::runtime::Backend::create(type);
    // });

    // /////
    // ///// Misc Methods
    // /////
    // //
    // // Methods that require the above types to be first declared.

    // mod.method("create_tensor", [](
    //     const std::shared_ptr<ngraph::runtime::Backend> backend,
    //     const ngraph::element::Type& element_type,
    //     const jlcxx::ArrayRef<int64_t> jl_shape,
    //     void* ptr)
    // {
    //     return backend->create_tensor(
    //         element_type,
    //         construct<ngraph::Shape>(jl_shape),
    //         ptr
    //     );
    // });

    /////
    ///// Ops
    /////

    mod.method(
        "op_add",
        [](const std::shared_ptr<ov::Node>& arg0, const std::shared_ptr<ov::Node>& arg1) {
            return makeop(Add, arg0, arg1);
        }
    );

    mod.method(
        "op_avgpool",
        [](const std::shared_ptr<ov::Node>& arg,
           const jlcxx::ArrayRef<int64_t> strides,
           const jlcxx::ArrayRef<int64_t> pads_begin,
           const jlcxx::ArrayRef<int64_t> pads_end,
           const jlcxx::ArrayRef<int64_t> kernel,
           bool exclude_pad) {
            return makeop(
                AvgPool,
                arg,
                construct<ov::Strides>(strides),
                construct<ov::Shape>(pads_begin),
                construct<ov::Shape>(pads_end),
                construct<ov::Shape>(kernel),
                exclude_pad,
                ov::op::RoundingType::FLOOR,
                ov::op::PadType::EXPLICIT
            );
        }
    );

    // // mod.method("op_batchnorm_training", [](
    // //     const std::shared_ptr<ngraph::Node> input,
    // //     const std::shared_ptr<ngraph::Node> gamma,
    // //     const std::shared_ptr<ngraph::Node> beta,
    // //     double epsilon)
    // // {
    // //     auto a = std::make_shared<ngraph::op::BatchNormTraining>(input,
    // gamma, beta, epsilon);
    // //     return std::dynamic_pointer_cast<ngraph::Node>(a);
    // // });

    mod.method(
        "op_broadcast",
        [](const std::shared_ptr<ov::Node>& arg,
           const std::shared_ptr<ov::Node>& target_shape) {
            return makeop(Broadcast, arg, target_shape);
        }
    );

    mod.method(
        "op_concat",
        [](const jlcxx::ArrayRef<std::shared_ptr<ov::Node>> jl_nodes, int64_t axis) {
            return makeop(Concat, construct<ov::OutputVector>(jl_nodes), axis);
        }
    );

    // Strategy for constants,
    // pass the julia array as an array of UInt8s - then we can use
    // reinterpret-cast to convert this to the type we want.
    mod.method(
        "op_constant",
        [](const ov::element::Type& type,
           const jlcxx::ArrayRef<int64_t> jl_shape,
           const jlcxx::ArrayRef<uint8_t> jl_values) {
            ov::Shape shape = construct<ov::Shape>(jl_shape);
            ov::element::Type_t type_enum = ov::element::Type_t(type);

            // TODO: Finish up constant construction
            switch (type_enum) {
            case ov::element::Type_t::f32:
                return makeop(
                    Constant,
                    type,
                    std::move(shape),
                    castcollect<float>(jl_values)
                );
            case ov::element::Type_t::f64:
                return makeop(
                    Constant,
                    type,
                    std::move(shape),
                    castcollect<double>(jl_values)
                );
            case ov::element::Type_t::i64:
                return makeop(
                    Constant,
                    type,
                    std::move(shape),
                    castcollect<int64_t>(jl_values)
                );
            default:
                throw std::runtime_error("Unsupported type");
            }
        }
    );

    mod.method(
        "op_convert",
        [](const std::shared_ptr<ov::Node>& arg, const ov::element::Type& element_type) {
            return makeop(Convert, arg, element_type);
        }
    );

    mod.method(
        "op_convolution",
        [](const std::shared_ptr<ov::Node>& data_batch,
           const std::shared_ptr<ov::Node>& filters,
           const jlcxx::ArrayRef<int64_t> strides,
           const jlcxx::ArrayRef<int64_t> pads_begin,
           const jlcxx::ArrayRef<int64_t> pads_end,
           const jlcxx::ArrayRef<int64_t> dilations) {
            return makeop(
                Convolution,
                data_batch,
                filters,
                construct<ov::Strides>(strides),
                construct<ov::CoordinateDiff>(pads_begin),
                construct<ov::CoordinateDiff>(pads_end),
                construct<ov::Strides>(dilations)
            );
        }
    );

    mod.method(
        "op_divide",
        [](const std::shared_ptr<ov::Node>& arg0, const std::shared_ptr<ov::Node>& arg1) {
            return makeop(Divide, arg0, arg1);
        }
    );

    // // mod.method("op_goe", [](
    // //     const std::shared_ptr<ov::Node>& arg,
    // //     uint64_t n)
    // // {
    // //     return makeop(v1::GetOutputElement, arg, n);
    // // });

    mod.method("op_log", [](const std::shared_ptr<ov::Node>& arg) {
        return makeop(Log, arg);
    });

    mod.method(
        "op_maximum",
        [](const std::shared_ptr<ov::Node>& arg0, const std::shared_ptr<ov::Node>& arg1) {
            return makeop(Maximum, arg0, arg1);
        }
    );

    // mod.method("op_maxpool", [](
    //     const std::shared_ptr<ov::Node>& arg,
    //     const jlcxx::ArrayRef<int64_t> strides,
    //     const jlcxx::ArrayRef<int64_t> pads_begin,
    //     const jlcxx::ArrayRef<int64_t> pads_end,
    //     const jlcxx::ArrayRef<int64_t> kernel)
    // {
    //     return makeop(
    //         MaxPool,
    //         arg,
    //         construct<ov::Strides>(strides),
    //         construct<ov::Shape>(pads_begin),
    //         construct<ov::Shape>(pads_end),
    //         construct<ov::Shape>(kernel),
    //         ov::op::RoundingType::FLOOR,
    //         ov::op::PadType::EXPLICIT
    //     );
    // });

    mod.method(
        "op_minimum",
        [](const std::shared_ptr<ov::Node>& arg0, const std::shared_ptr<ov::Node>& arg1) {
            return makeop(Minimum, arg0, arg1);
        }
    );

    mod.method(
        "op_matmul",
        [](const std::shared_ptr<ov::Node>& A,
           const std::shared_ptr<ov::Node>& B,
           bool transpose_a,
           bool transpose_b) { return makeop(MatMul, A, B, transpose_a, transpose_b); }
    );

    mod.method(
        "op_mul",
        [](const std::shared_ptr<ov::Node>& arg0, const std::shared_ptr<ov::Node>& arg1) {
            return makeop(Multiply, arg0, arg1);
        }
    );

    mod.method("op_negative", [](const std::shared_ptr<ov::Node>& arg) {
        return makeop(Negative, arg);
    });

    mod.method(
        "op_parameter",
        [](const ov::element::Type& element_type, const jlcxx::ArrayRef<int64_t, 1> shape) {
            return makeop(Parameter, element_type, construct<ov::Shape>(shape));
        }
    );

    mod.method(
        "op_power",
        [](const std::shared_ptr<ov::Node>& arg0, const std::shared_ptr<ov::Node>& arg1) {
            return makeop(Power, arg0, arg1);
        }
    );

    mod.method("op_relu", [](const std::shared_ptr<ov::Node>& arg) {
        return makeop(Relu, arg);
    });

    mod.method("op_reshape", [](
        const std::shared_ptr<ov::Node>& arg,
        const std::shared_ptr<ov::Node>& shape_pattern)
    {
        return makeop(Reshape, arg, shape_pattern, false);
    });

    mod.method("op_sigmoid", [](const std::shared_ptr<ov::Node>& arg) {
        return makeop(Sigmoid, arg);
    });

    // // mod.method("op_slice", [](
    // //     const std::shared_ptr<ov::Node>& arg,
    // //     const jlcxx::ArrayRef<int64_t,1> lower_bounds,
    // //     const jlcxx::ArrayRef<int64_t,1> upper_bounds)
    // // {
    // //     return makeop(
    // //         v0::Slice,
    // //         arg,
    // //         construct<ov::Coordinate>(lower_bounds),
    // //         construct<ov::Coordinate>(upper_bounds)
    // //     );
    // // });

    mod.method("op_softmax", [](const std::shared_ptr<ov::Node>& arg, int64_t axis) {
        return makeop(Softmax, arg, axis);
    });

    mod.method("op_sqrt", [](const std::shared_ptr<ov::Node>& arg) {
        return makeop(Sqrt, arg);
    });

    mod.method(
        "op_subtract",
        [](const std::shared_ptr<ov::Node>& arg0, const std::shared_ptr<ov::Node>& arg1) {
            return makeop(Subtract, arg0, arg1);
        }
    );

    mod.method("op_reducesum", [](
        const std::shared_ptr<ov::Node>& arg,
        const std::shared_ptr<ov::Node>& reduction_axes)
    {
        return makeop(ReduceSum, arg, reduction_axes);
    });
}
