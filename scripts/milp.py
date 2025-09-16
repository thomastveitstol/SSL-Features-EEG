import time

import cvxpy
import numpy.random


def main():
    start_time = time.time()

    sample_size = 50
    tot_num_experiments = 1000
    big_m = 1000

    X = numpy.random.normal(0, 5, size=(sample_size, tot_num_experiments))
    y = numpy.random.uniform(0, 100, size=(sample_size,))

    # Inputs
    # Example data shapes (replace with your actual data)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    group_size = tot_num_experiments // 5
    num_groups = 5
    assert n_features == group_size * num_groups

    # Define variables
    theta = cvxpy.Variable(n_features)  # Actual regression coefficients used
    z = cvxpy.Variable(n_features, boolean=True)  # Binary indicators for selection

    # Group constraints: only one feature per group
    constraints = []
    for g in range(num_groups):
        start = g * group_size
        end = (g + 1) * group_size

        # Only one feature per group (sum should be one)
        constraints.append(cvxpy.sum(z[start:end]) == 1)

        # Big M constraint
        constraints += ([theta[j] <= big_m * z[j] for j in range(start, end)]
                        + [theta[j] >= -big_m * z[j] for j in range(start, end)])

    # Objective: minimize squared error
    objective = cvxpy.Minimize(cvxpy.sum_squares(X @ theta - y))

    # Problem definition
    problem = cvxpy.Problem(objective, constraints)

    # Solve
    problem.solve(solver=cvxpy.SCIP, verbose=True)

    # Output
    selected_features = [j for j in range(n_features) if z.value[j] > 0.5]
    selected_betas = theta.value[selected_features]

    print("Selected features:", selected_features)
    print("Corresponding coefficients:", selected_betas)
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time):.2f} seconds")


if __name__ == "__main__":
    main()
