import pickle
import numpy as np
import time

with open('data.pkl', 'rb') as f:
  data = pickle.load(f)
training_data, test_data = data[0], data[2]

np.random.seed( 1000 )

n_input, n_hidden, n_output = 784, 32, 10
biases = [ np.random.randn(n_hidden, 1), np.random.randn(n_output, 1)]
weights = [ np.random.randn(n_hidden, n_input), np.random.randn(n_output, n_hidden)]

n_epochs, lr = 1000, 5 / 50000

def sigmoid(z, deriv = False):
  if not deriv:
    return 1 / (1 + np.exp(-z))
  return sigmoid(z) * (1 - sigmoid(z))

def forward(x):
  wxb0 = np.matmul(weights[0], x) + biases[0]
  hidden = sigmoid(wxb0)
  wxb1 = np.matmul(weights[1], hidden) + biases[1]
  output = sigmoid(wxb1)
  return wxb0, hidden, wxb1, output
  
def backprop(x, y):
  nabla_b = [ np.zeros(biases[0].shape), np.zeros(biases[1].shape) ]
  nabla_w = [ np.zeros(weights[0].shape), np.zeros(weights[1].shape) ]
  
  # forward pass
  wxb0, hidden, wxb1, output = forward( x )

  # backward pass
  nabla_b[1] = (output - y) * sigmoid(wxb1, True)
  nabla_w[1] = np.outer(nabla_b[1], hidden)
  nabla_b[0] = np.dot(weights[1].T, nabla_b[1]) * sigmoid(wxb0, True)
  nabla_w[0] = np.outer(nabla_b[0], x)
  return nabla_w, nabla_b

for ep in range(n_epochs):
  print('epoch: ', ep)

  t = time.time()

  # train
  nabla_w = [ np.zeros(weights[0].shape), np.zeros(weights[1].shape) ]
  nabla_b = [ np.zeros(biases[0].shape), np.zeros(biases[1].shape) ]

  for x, y in training_data:
    nabla_wi, nabla_bi = backprop(x, y)
    nabla_w = [ nw + nwi for nw, nwi in zip(nabla_w, nabla_wi) ]
    nabla_b = [ nb + nbi for nb, nbi in zip(nabla_b, nabla_bi) ]

  weights = [ w - lr * nw for w, nw in zip(weights, nabla_w) ]
  biases = [ b - lr * nb for b, nb in zip(biases, nabla_b) ]

  print('time = ', time.time() - t)
  
  # evaluate
  s = 0
  for x, y in test_data:
    _, _, _, output = forward( x )
    s += int(np.argmax(output) == y)
  print("Epoch {} : {} / {}".format( ep, s, len(test_data) ))


