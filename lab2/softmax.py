import numpy as np

from lab2.activations import softmax


class SoftmaxClassifier:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.W = None
        self.initialize()

    def initialize(self):
        # initialize the weight matrix (remember the bias trick) with small random variables
        # you might find np.random.randn useful here *0.001
        self.W = np.random.rand(self.num_classes, self.input_shape) * 0.001
        # sucht that it is C X 3073 where 3073 = 32x32x3 + 1 (so an image sample)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X
        dot_prod = np.dot(self.W, np.transpose(X))
        # remember about the bias trick!
        # 1. apply the softmax function on the scores
        softmax_scores = softmax(dot_prod)
        # 2, returned the scores
        return softmax_scores

    def predict(self, X: np.ndarray) -> int:
        label = None
        # TODO your code here
        # 0. compute the dot product between the weight matrix and the input X as the scores
        dot_prod = np.dot(self.W, np.transpose(X))
        # 1. compute the prediction by taking the argmax of the class scores
        label = np.argmax(dot_prod, axis=0)
        return label

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs) -> dict:

        history = []

        bs = kwargs['bs'] if 'bs' in kwargs else 128
        reg_strength = kwargs['reg_strength'] if 'reg_strength' in kwargs else 1e3
        steps = kwargs['steps'] if 'steps' in kwargs else 100
        lr = kwargs['lr'] if 'lr' in kwargs else 1e-3
        print('Fit with: bs={}, reg_strength={}, steps={}, lr={}'.format(bs, reg_strength, steps, lr))

        # run mini-batch gradient descent
        for iteration in range(0, steps):
            # TODO your code here
            # sample a batch of images from the training set
            # you might find np.random.choice useful
            indices = np.random.choice(range(0, np.shape(X_train)[0]), size=bs)
            # 0..50000 select bs=200 random indices (values)
            X_batch, y_batch = X_train[indices], y_train[indices]
            # x_batch should have the shape (bs, 3072)

            logits = np.dot(self.W, np.transpose(X_batch))  # shape: (C x bs)
            logits -= np.max(logits)  # to avoid the -inf issue
            exponential_sum = np.sum(np.exp(logits))

            # now compute the partial derivative of W to update W
            oneHotEncoding = self.npToOneHot(y_batch)
            CT = softmax(np.dot(X_batch, np.transpose(self.W))) - oneHotEncoding
            # something - oneHotEncoding, where:
            # - something is (bs x 3073) dot (3073 x C) = (bs x C)
            # - oneHotEncoding has size (bs x C)
            # so the shapes of the elements to be subtracted match
            dW = np.transpose(np.dot(np.transpose(X_batch), CT))
            # dw should have the same shape as W: (C x 3073) = [(3073 X bs) dot (bs X C)] everything transposed
            regularization = reg_strength * np.sum(self.W)  # np.square()
            # loss = np.sum(np.subtract(y_train, exponential_sum)) / bs + regularization
            loss = np.sum(np.subtract(np.log(exponential_sum), y_train)) / bs + regularization
            # we take all ground truth labels, subtract the exponential sum and sum everything up

            # loss explained:
            # loss_i = -log(softmax(sample)) // trick with subtracting the max
            # loss_i = log(sum) - log(exp(f_yi)) // use this form to compute
            # loss_i = log(sum) - f_yi

            # end TODO your code here
            # perform a parameter update
            self.W += - lr * dW  # 10x3073
            # append the training loss to the history dict
            history.append(loss)

        return history

    def npToOneHot(self, arr):
        arr_zeros = np.zeros((arr.size, arr.max() + 1))
        arr_zeros[np.arange(arr.size), arr] = 1
        return arr_zeros

    def get_weights(self, img_shape):
        W = None
        # TODO your code here
        # 0. ignore the bias term
        weights_without_bias = self.W[:, 0:-1]
        # 1. reshape the weights to (*image_shape, num_classes)
        W = np.reshape(weights_without_bias, (self.num_classes, *img_shape))
        return W

    def load(self, path: str) -> bool:
        # TODO your code here
        # load the input shape, the number of classes and the weight matrix from a file
        [self.input_shape, self.num_classes, self.W] = np.load('saved_classifier_data.npy', allow_pickle=True)
        return True

    def save(self, path: str) -> bool:
        # TODO your code here
        # save the input shape, the number of classes and the weight matrix to a file
        # you might find np.save useful for this
        np.save('saved_classifier_data', [self.input_shape, self.num_classes, self.W])
        # TODO your code here

        return True

