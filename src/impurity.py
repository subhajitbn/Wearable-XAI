def gini_impurity(p):
    """Gini impurity for a Bernoulli node with class-1 rate p"""
    return 2 * p * (1 - p)

def impurity_reduction(pL, nL, pR, nR):
    """
    Compute impurity reduction given child probabilities and supports.
    Parameters:
        pL (float): class-1 rate in left child
        nL (int): number of samples in left child
        pR (float): class-1 rate in right child
        nR (int): number of samples in right child
    Returns:
        float: impurity reduction (ΔG)
    """
    N = nL + nR
    # parent rate
    p = (pL * nL + pR * nR) / N
    # parent impurity
    G_parent = gini_impurity(p)
    # children impurity
    G_children = (nL / N) * gini_impurity(pL) + (nR / N) * gini_impurity(pR)
    # reduction
    return G_parent - G_children

if __name__ == "__main__":
    # Example: pL=0, nL=39, pR=0.983, nR=59
    print(impurity_reduction(0, 39, 0.983, 59))
