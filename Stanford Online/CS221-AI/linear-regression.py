import numpy  as np

trainExamples = [[1, 1], [2, 3], [4, 3]]

# initial weight vector is all zeros vector with dimension 2
initialWeight = np.array([0, 0])

# the feature vector
def phi(x):
    return np.array([1, x])

# equation in note
def trainLoss(w):
    return 1/len(trainExamples) * sum((w.dot(phi(x)) - y) ** 2 for x, y in trainExamples)

def gradientTrainLoss(w):
    return 1/len(trainExamples) * sum(2 * (w.dot(phi(x)) - y) * phi(x) for x, y in trainExamples) 

def gradientDescent(gradientFunction, weight, stepSize, lossFunction):
    for i in range(500):
        gradient = gradientFunction(weight)
        weight = weight - stepSize * gradient
        print(f"epoch {i}: loss = {lossFunction(weight)}, weight = {weight}")

gradientDescent(gradientTrainLoss, initialWeight, 0.1, trainLoss)


# In NumPy, the choice between * and .dot() completely changes the math you are performing.
# 1. Element-Wise Multiplication (*)
#    The * operator performs element-wise multiplication (also called the Hadamard product). It multiplies the first element of list A by the first of list B, the second by the second, and so on.
#    Output: Returns an array of the same shape as your inputs.Example: [1, 2] * [3, 4] results in [3, 8].
# 2. Dot Product (.dot() or @)
#    The .dot() method (or the @ operator in newer Python) performs matrix/vector multiplication. It multiplies corresponding elements and then sums them all up.
#    Output: Returns a single number (a scalar) when used on two vectors.Example: np.dot([1, 2], [3, 4]) results in (1*3) + (2*4) = 11.