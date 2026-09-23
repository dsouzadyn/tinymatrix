"""PageRank algorithm implementation using TinyMatrix."""

from tinymatrix import Matrix


def compute_pagerank(links, d=0.85, max_iter=100, tol=1e-6):
    """Compute PageRank scores using power iteration on Markov transition matrix."""
    n = len(links)
    # Transition probability matrix M[j, i] = probability of going from i to j
    M = Matrix.zeroes(n, n)

    for i, outgoing in enumerate(links):
        if outgoing:
            prob = 1.0 / len(outgoing)
            for j in outgoing:
                M.M[j][i] = prob
        else:
            # Dangling node jumps equally to all nodes
            for j in range(n):
                M.M[j][i] = 1.0 / n

    # Teleportation matrix E
    E = Matrix.ones(n, n) * (1.0 / n)

    # Google matrix G = d * M + (1 - d) * E
    G = M * d + E * (1.0 - d)

    # Initial uniform rank vector
    v = Matrix.ones(n, 1) * (1.0 / n)

    for iteration in range(max_iter):
        v_next = G @ v
        diff = (v_next - v).norm(ord=1)
        v = v_next
        if diff < tol:
            print(f"PageRank converged in {iteration + 1} iterations.")
            break

    return v.flatten()


if __name__ == "__main__":
    # Web graph with 4 pages:
    # 0 links to 1, 2
    # 1 links to 2
    # 2 links to 0
    # 3 links to 2
    graph = [
        [1, 2],
        [2],
        [0],
        [2],
    ]
    scores = compute_pagerank(graph)
    print("PageRank scores:")
    for idx, score in enumerate(scores):
        print(f"  Page {idx}: {score:.4f}")
