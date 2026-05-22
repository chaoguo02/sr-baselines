def apply_rag_sr_result_schema(
    result,
    *,
    best_generation=0,
    final_generation=0,
    n_logged_generations=1,
    expression_key="best_expression",
):
    """
    Add a stable set of RAG-SR-style aliases to baseline run summaries so that
    downstream plotting/aggregation can use one schema across methods.
    """
    normalized = dict(result or {})

    train_mse = normalized.get("train_mse")
    test_mse = normalized.get("test_mse")
    best_expression = normalized.get(expression_key) or normalized.get("best_expression")

    normalized.setdefault("train_fitness", train_mse)
    normalized.setdefault("test_fitness", test_mse)
    normalized.setdefault("expression", best_expression)

    normalized.setdefault("best_train_fitness", train_mse)
    normalized.setdefault("best_test_fitness", test_mse)
    normalized.setdefault("final_train_fitness", train_mse)
    normalized.setdefault("final_test_fitness", test_mse)

    normalized.setdefault("best_generation", int(best_generation))
    normalized.setdefault("final_generation", int(final_generation))
    normalized.setdefault("n_logged_generations", int(n_logged_generations))
    normalized.setdefault("final_expression", best_expression)

    return normalized
