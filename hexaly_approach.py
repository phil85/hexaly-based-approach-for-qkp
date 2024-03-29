import numpy as np
import pandas as pd
import localsolver


def compute_ofv(items, nodes, edges):

    # Compute a dense utility matrix
    utility_matrix = np.zeros((len(nodes), len(nodes)))
    rows, cols = np.array(list(edges.keys())).T
    values = np.array(list(edges.values()))
    utility_matrix[rows, cols] = values
    utility_matrix[cols, rows] = values

    # Set diagonal elements to zero
    linear_utilities = np.diagonal(utility_matrix).copy()
    utility_matrix[np.diag_indices_from(utility_matrix)] = 0

    # Add linear utilities
    items = np.array(items, dtype=int)
    ofv = linear_utilities[items].sum()

    # Add quadratic utilities
    ofv += utility_matrix[items][:, items].sum() / 2

    return ofv


def run_hexaly_approach(nodes, edges, weights, budgets, params):

    # Initialize results
    results = pd.DataFrame()

    # Set time limit
    if 'time_limit' in params:
        time_limit = params['time_limit']
    else:
        time_limit = 1e10

    for budget in budgets:

        with localsolver.LocalSolver() as ls:

            # Number of items
            nb_items = len(nodes)

            # Declare the optimization model
            model = ls.model

            # Parameterize the solver
            ls.param.time_limit = time_limit
            ls.param.verbosity = 0

            # Decision variables x[i]
            x = [model.bool() for i in range(nb_items)]

            # Weight constraint
            knapsack_weight = model.sum(x[i] * weights[i] for i in range(nb_items))
            model.constraint(knapsack_weight <= budget)

            # Maximize value
            knapsack_value = model.sum(x[i] * x[j] * edges[i, j] for i, j in edges)
            model.maximize(knapsack_value)

            model.close()

            ls.solve()

            # Get results
            result = pd.Series(dtype=object)

            try:
                result['items'] = [i for i in nodes if x[i].value > 0.5]
                result['cpu'] = ls.get_statistics().get_running_time()
                result['mip_gap'] = ls.get_solution().get_objective_gap(0)
            except:
                result['items'] = np.array([], dtype=int)
                result['cpu'] = time_limit
                result['mip_gap'] = -1

        result['ofv'] = compute_ofv(result['items'], nodes, edges)
        result['budget'] = budget
        result['budget_fraction'] = '{:.4f}'.format(budget / sum(weights))
        result['total_weight'] = sum([weights[i] for i in result['items']])
        result['approach'] = 'hexaly'

        # Convert to dataframe
        result = result.to_frame().transpose()

        # Append results to result
        results = pd.concat((results, result))

    return results
