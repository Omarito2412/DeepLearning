#### MNIST results using Neural Network with different parameters
In this document I will gather the results I achieved using a simple neural network on the MNIST dataset.

The first result is where I took the network a little bit too far using a hidden layer of 800 neurons and training it for 150 epochs, it performed a not bad accuracy but still very low for the huge hidden layer.
![800 Neurons, sigmoid]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/800n150ep_curve.png)
![800 Neurons result]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/800n150ep.png)

50 Neurons, No regularization, 200 epochs, 0.3 learning rate, 30% Cross-validation:

##### Sigmoid activation:
![Sigmoid Activation curve]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_sigcurve.png)
![Sigmoid Activation result]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n_200ep_sigmoid.png)
##### Tanh activation:
![Tanh Activation curve]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_tanhcurve.png)
![Tanh Activation result]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_tanh.png)
##### Leaky ReLU:
![ReLU Activation curve]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_relucurve.png)
![ReLU Activation result]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_relu.png)

### Dropout
Testing the same network using Dropout and the 3 different activations
##### Sigmoid activation:
![Sigmoid Activation curve]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_sig_drop_curve.png)
![Sigmoid Activation result]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_sig_drop.png)
##### Tanh activation:
![Tanh Activation curve]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_tanh_drop_curve.png)
![Tanh Activation result]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_tanh_drop.png)
##### Leaky ReLU:
![ReLU Activation curve]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_relu_drop_curve.png)
![ReLU Activation result]
(https://github.com/Omarito2412/DeepLearning/blob/master/MLP/benchmarks/50n200ep_relu_drop.png)
