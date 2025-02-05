def _exchange_current_density_inputs(variable_names):
    j0_n_input = "j0_n" in variable_names
    j0_p_input = "j0_p" in variable_names
    alpha_n_input = "alpha_n" in variable_names
    alpha_p_input = "alpha_p" in variable_names
    return j0_n_input, j0_p_input, alpha_n_input, alpha_p_input
