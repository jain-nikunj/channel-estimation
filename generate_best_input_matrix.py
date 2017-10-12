import numpy as np
import numpy.linalg as la
from random import random
from array import array

def generateMatrix(numSamples=10**3, taps=70, iterations=1000,
  costFunction = lambda singularValues: sum([1/value**2 for value in singularValues])):
  """Takes in the number of estimated taps the channel has, as well as the
  number of iterations to repeat in matrix estimation. In each simulation,
  generates a random sequence of numSamples which are all either +1 or -1. Based
  on this, generates the matrix of a taps-delayed filter. For each matrix,
  calculates a cost function, and chooses the matrix which minimizes the cost
  function.
  """
  bestSeq, lowestCost = [], float("inf")

  for _ in range(iterations):
    sequence = [1 if random() > 0.5 else -1 for sample in range(numSamples)]
    matrix = []
    for i in range((numSamples - taps)):
      matrix.append(sequence[i: i + taps])

    singularValues = la.svd(matrix, compute_uv = 0)
    cost = costFunction(singularValues)
    if cost < lowestCost:
      lowestCost = cost
      bestSeq = sequence

  return bestSeq, lowestCost

with open('sequence', 'wb') as outFile:
  seq, loss = generateMatrix()
  intArray = array('i', seq)
  intArray.tofile(outFile)
