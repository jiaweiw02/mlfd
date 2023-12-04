def L(k, x):
    if k == 0:
        return 1
    if k == 1:
        return x

    eq1 = ((2 * k - 1) / k) * x
    eq2 = (k - 1) / k

    return eq1 * L(k - 1, x) - eq2 * L(k - 2, x)


def eighthOrderTransform(x1, x2):
    return [1,
            L(1, x1), L(1, x2),

            L(2, x1), L(1, x1) * L(1, x2), L(2, x2),

            L(3, x1), L(2, x1) * L(1, x2), L(1, x1) * L(2, x2), L(3, x2),

            L(4, x1), L(3, x1) * L(1, x2), L(2, x1) * L(2, x2),
            L(1, x1) * L(3, x2), L(4, x2),

            L(5, x1), L(4, x1) * L(1, x2), L(3, x1) * L(2, x2),
            L(2, x1) * L(3, x2), L(1, x1) * L(4, x2), L(5, x2),

            L(6, x1), L(5, x1) * L(1, x2), L(4, x1) * L(2, x2),
            L(3, x1) * L(3, x2), L(2, x1) * L(4, x2), L(1, x1) * L(5, x2),
            L(6, x2),

            L(7, x1), L(6, x1) * L(1, x2), L(5, x1) * L(2, x2),
            L(4, x1) * L(3, x2), L(3, x1) * L(4, x2), L(2, x1) * L(5, x2),
            L(1, x1) * L(6, x2), L(7, x2),

            L(8, x1), L(7, x1) * L(1, x2), L(6, x1) * L(2, x2),
            L(5, x1) * L(3, x2), L(4, x1) * L(4, x2), L(3, x1) * L(5, x2),
            L(2, x1) * L(6, x2), L(1, x1) * L(7, x2), L(8, x2)]