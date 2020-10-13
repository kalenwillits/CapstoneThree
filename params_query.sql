SELECT DISTINCT gate, weight_mod, window, epochs, vector_scope, vector_weight
FROM param_test
ORDER BY accuracy DESC, precision DESC
LIMIT 1;
