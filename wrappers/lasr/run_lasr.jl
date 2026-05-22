using DelimitedFiles
using JSON
using LibraryAugmentedSymbolicRegression

function map_binary_operator(name::String)
    ops = Dict(
        "+" => (+),
        "-" => (-),
        "*" => (*),
        "/" => (/),
        "^" => (^),
    )
    haskey(ops, name) || error("Unsupported binary operator: $name")
    return ops[name]
end

function map_unary_operator(name::String)
    ops = Dict(
        "sin" => sin,
        "cos" => cos,
        "exp" => exp,
        "log" => log,
        "sqrt" => sqrt,
        "abs" => abs,
    )
    haskey(ops, name) || error("Unsupported unary operator: $name")
    return ops[name]
end

function mse(y_true, y_pred)
    diffs = y_true .- y_pred
    return sum(diffs .* diffs) / length(diffs)
end

function main()
    cfg_path = ARGS[1]
    cfg = JSON.parsefile(cfg_path)
    api_key = get(ENV, cfg["api_key_env"], "")

    train = Array{Float64}(readdlm(cfg["train_csv"], ',', Float64))
    test = Array{Float64}(readdlm(cfg["test_csv"], ',', Float64))

    x_train = train[:, 1:end-1]
    y_train = vec(train[:, end])
    x_test = test[:, 1:end-1]
    y_test = vec(test[:, end])

    X = permutedims(x_train)
    binary_operators = map(map_binary_operator, cfg["binary_operators"])
    unary_operators = map(map_unary_operator, cfg["unary_operators"])
    llm_p = Float64(cfg["llm_probability"])

    options = LaSROptions(;
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=Int(cfg["populations"]),
        use_llm=Bool(cfg["use_llm"]),
        use_concepts=Bool(cfg["use_concepts"]),
        use_concept_evolution=Bool(cfg["use_concept_evolution"]),
        llm_operation_weights=LLMOperationWeights(;
            llm_crossover=llm_p, llm_mutate=llm_p, llm_randomize=llm_p
        ),
        num_generated_equations=Int(cfg["num_generated_equations"]),
        num_generated_concepts=Int(cfg["num_generated_concepts"]),
        num_pareto_context=Int(cfg["num_pareto_context"]),
        max_concepts=Int(cfg["max_concepts"]),
        is_parametric=Bool(cfg["is_parametric"]),
        llm_context=cfg["llm_context"],
        variable_names=Dict{String,String}(k => v for (k, v) in cfg["variable_names"]),
        prompts_dir=cfg["prompts_dir"],
        api_key=api_key,
        model=cfg["model_name"],
        api_kwargs=Dict("url" => cfg["api_url"], "max_tokens" => Int(cfg["api_max_tokens"])),
        verbose=Bool(cfg["verbose"]),
        seed=Int(cfg["seed"]),
    )

    parallelism = Symbol(cfg["parallelism"])
    start_time = time()
    hall_of_fame = equation_search(
        X,
        y_train;
        niterations=Int(cfg["niterations"]),
        options=options,
        parallelism=parallelism,
    )
    runtime_seconds = time() - start_time

    dominating = calculate_pareto_frontier(hall_of_fame)
    length(dominating) > 0 || error("No equations found by LaSR.")

    best_member = dominating[1]
    for member in dominating
        if member.loss < best_member.loss
            best_member = member
        end
    end

    train_pred, train_ok = eval_tree_array(best_member.tree, X, options)
    train_ok || error("Failed to evaluate best LaSR expression on train data.")
    test_pred, test_ok = eval_tree_array(best_member.tree, permutedims(x_test), options)
    test_ok || error("Failed to evaluate best LaSR expression on test data.")

    best_expression = string_tree(best_member.tree, options)
    complexity = compute_complexity(best_member, options)

    pareto = [
        Dict(
            "expression" => string_tree(member.tree, options),
            "loss" => member.loss,
            "complexity" => compute_complexity(member, options),
        ) for member in dominating
    ]

    result = Dict(
        "algorithm" => "LaSR",
        "function_id" => Int(cfg["function_id"]),
        "function_name" => cfg["function_name"],
        "n_variables" => Int(cfg["n_variables"]),
        "train_mse" => mse(y_train, train_pred),
        "test_mse" => mse(y_test, test_pred),
        "runtime_seconds" => runtime_seconds,
        "best_expression" => best_expression,
        "best_loss" => best_member.loss,
        "best_complexity" => complexity,
        "pareto_frontier" => pareto,
        "status" => "success",
    )

    open(joinpath(cfg["output_dir"], "run_summary.json"), "w") do io
        JSON.print(io, result, 2)
    end
end

main()
