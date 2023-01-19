def _fmt_variables(variables):
    lst = []
    for v in variables:
        var = v.__dict__.copy()

        var["prior"] = var["prior"].__dict__
        if "_boundaries" in var["prior"].keys():
            var["prior"] = var["prior"]["_boundaries"].__dict__.copy()
        lst.append(var)
    return lst


def _fmt_parameters(parameters):
    return {k: str(v) for k, v in parameters.items()}
