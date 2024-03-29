from hexaly_approach import run_hexaly_approach

# Define knapsack capacity (or budget)
budgets = [8]

# Define items (or nodes) with ids 0, 1, ..., n
items = [0, 1, 2, 3]

# Define weights for items
weights = [2, 3, 4, 5]

# Define profits for including items i and j in the knapsack
profits = {(0, 0): 1,
           (0, 1): 2,
           (0, 2): 11,
           (1, 1): 1,
           (1, 2): 3,
           (1, 3): 2,
           (2, 2): 1,
           (3, 3): 1}

# Define parameters
params = {'time_limit': 60}

# Run the breakpoints algorithm
results = run_hexaly_approach(items, profits, weights, budgets, params)

# Print results
print('Objective function value: {:.1f}'.format(results.loc[0, 'ofv']))
print('Selected items: {:}'.format(results.loc[0, 'items']))
