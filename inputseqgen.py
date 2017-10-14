import numpy as np
import numpy.linalg as la
from random import random
from array import array

M = 128
N = 1
ITER = 1000

#Contains original sequence of lenght M + N - 1 (In time order)
FILENAMEORIGINAL = 'dumpfiles/filesource_m'+str(M)+'_n' + str(N)+'_original'

#Contains 3 copies of input sequence (since bytes missing in beginning and end of output stream)
FILENAMEREPEAT = 'dumpfiles/filesource_m'+str(M)+'_n' + str(N)+'_repeat'

#To prevent divide by 0 warnings
eps = 1e-10

def generateMatrix(numSamples=126, taps=3, iterations=1000,
  costFunction = lambda singularValues: sum([1/(value+eps)**2 for value in singularValues])):
  """Takes in the number of estimated taps the channel has, as well as the
  number of iterations to repeat in matrix estimation. In each simulation,
  generates a random sequence of numSamples which are all either +1 or -1. Based
  on this, generates the matrix of a taps-delayed filter. For each matrix,
  calculates a cost function, and chooses the matrix which minimizes the cost
  function.
  """
  bestSeq, lowestCost = [], float("inf")

  for _ in range(iterations):
    sequence = [1 if random() > 0.5 else -1 for sample in range(numSamples+taps -1)]
    matrix = []
    for i in range((numSamples)):
      matrix.append([])
      for j in range((taps)):
        matrix[i].append(sequence[i+taps-1-j])

    singularValues = la.svd(matrix, compute_uv = 0)
    cost = costFunction(singularValues)
    if cost < lowestCost:
      lowestCost = cost
      bestSeq = sequence
      # print bestSeq
      # for line in matrix:
      #   print(line)

  return bestSeq, lowestCost

def main():
  seq, loss = generateMatrix(M,N,ITER)
  print("Minimum error rate: " + str(loss))  
  # print(seq)

  #Convert sequences to 0s and 1s
  bitseq = [1 if s is 1 else 0 for s in seq]
  with open(FILENAMEORIGINAL, 'wb') as newFile:
    newFileByteArray = bytearray(bitseq)
    newFile.write(newFileByteArray)
    newFile.close()

  # print bitseq

  #Repeated sequence
  bitseqrepeat = []
  for i in range(3):
    for bits in bitseq:
      bitseqrepeat.append(bits)
  # print(bitseqrepeat)

  with open(FILENAMEREPEAT, 'wb') as newFile:
    newFileByteArrayRepeat = bytearray(bitseqrepeat)
    newFile.write(newFileByteArrayRepeat)
    newFile.close()



if __name__ == "__main__":
    main()