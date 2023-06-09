import numpy as np

def gini_algorithm(target_gini, num_categories):

    print("Start gini-process")
    # target_gini is the gini to be reached with the eventual distribution, num_categories is the number of categories in the eventual distribution
    # Define the gini function
    def gini(x):
        total = 0
        for i, xi in enumerate(x[:-1], 1):
            total += np.sum(np.abs(xi - x[i:]))
        return total / (len(x) ** 2 * np.mean(x))

    # Set the initial lower and upper bounds for the first element
    lower_bound = 0
    upper_bound = 100

    # Use binary search to find the appropriate value for the first element
    while True:
        mid = (lower_bound + upper_bound) / 2
        x = np.array([mid] + [1] * (num_categories - 1))
        g = gini(x)

        if abs(g - target_gini) < 1e-3:
            distribution = x / np.sum(x)
            print("Relative distribution over num_categories:", distribution)
            g = gini(distribution)
            print("Gini coefficient:", g)
            return g, distribution

        elif g > target_gini:
            upper_bound = mid
        else:
            lower_bound = mid


#target_gini = 0.5
#num_categories = 10
#g, r = gini_algorithm(target_gini, n)


