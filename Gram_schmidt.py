import numpy as np


##############################################################################
#                                   Functions
# ----------*----------*----------*----------*----------*----------*----------*
def proj(x, u):
    ## Can't hurt
    u = unit_vec(u)
    return np.dot(x, u) * u


def unit_vec(x):
    """Get unit vector of x. Same direction, norm==1"""
    return x / np.linalg.norm(x)


def gramSchmidt(vectors):


    ###### Ensure the input is a 2d array (or can be treated like one)
    vectors = np.atleast_2d(vectors)

    ###### Handle recursion end conditions
    if len(vectors) == 0:
        return []

    if len(vectors) == 1:
        return unit_vec(vectors)

    u = vectors[-1]

    ###### Orthonormalize the rest of the vector space
    basis = gramSchmidt(vectors[0:-1])

    ## Append this vector orthonormalized to the rest to the basis
    w = np.atleast_2d(u - np.sum(proj(u, v) for v in basis))
    basis = np.append(basis, unit_vec(w), axis=0)

    return basis


def modifiedGramSchmidt(vectors):
    """ _correct_ recursive implementation of Gram Schmidt algo that is not subject to
    rounding erros that the original formulation is.
    Function signature and usage is the same as gramSchmidt()
    """
    ###### Ensure the input is a 2d array (or can be treated like one)
    vectors = np.atleast_2d(vectors)

    ###### Handle End Conditions
    if len(vectors) == 0:
        return []

    ## Always just take unit vector of first vector for the start of the basis
    u1 = unit_vec(vectors[0])

    if len(vectors) == 1:
        return u1


    basis = np.vstack((u1, modifiedGramSchmidt(
        list(map(lambda v: v - proj(v, u1), vectors[1:])))))  # not explicit list(map) conversion, need for python3+

    return np.array(basis)


def _is_orthag(vectors):
    """Simple check, sees if all of the vectors in v are orthagonal to eachother.
    Takes the dot product of each vector pair, sees if the result is close to zero
    """
    orthag = True
    vectors = np.atleast_2d(vectors)
    for vector in vectors:
        for vector2 in vectors:
            ## Don't dot itself
            if np.array_equal(vector, vector2):
                continue
            ## Dot product alwys has some numerical precision remainder
            if abs(np.dot(vector, vector2)) > 1e-5:
                orthag = False
    return orthag0


def test():
    vectors = [[4,1,3,-1], [2,1,-3,4],[1,0,-2,7],[6,2,9,-5]]
    print("Test 1: Finding orthonormal basis for simple vector space {}".format(vectors))
    ## Example vectors from https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    ospace = gramSchmidt(vectors)
    ospace2 = modifiedGramSchmidt(vectors)
    ## Are the two about the same?
    samesies = np.array_equal(ospace, ospace2)
    print(
        "\nFor vector space provided {}\n\tOrthonormalized basis produced with classical Gram Schmidt process \n{}".format(
            vectors, ospace))
    print(
        "\n\tOrthonormal basis produced with Modified Gram Schmidt process: \n{}, \nAre the two about equal?? {}".format(
            ospace2, samesies))
    print("\tAre all basis vectors orthagonal to eachother? {}".format(_is_orthag(ospace)))

    vectors = [[3, 13, 2, 5], [1, 1, 2, 2], [8, -1, -0.5, 0], [1, -9, 0, 0]]
    print("\n\nTest 2: Finding orthonormal basis for arbritrary and more complex vector space {}".format(vectors))
    ospace = modifiedGramSchmidt(vectors)
    ## Are the two about the same?
    print(
        "\tFor vector space provided \n{}\nOrthonormalized basis produced with modified Gram Schmidt process \n{}".format(
            vectors, ospace))
    print("\tAre all basis vectors orthagonal to eachother? {}".format(_is_orthag(ospace)))


##############################################################################
#                              Runtime Execution
# ----------*----------*----------*----------*----------*----------*----------*
if __name__ == "__main__":
    test()