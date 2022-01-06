import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lab2.activations import softmax
from lab2.cifar10 import LABELS
from lab2.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from lab2.softmax import SoftmaxClassifier
from functools import reduce

TRAIN = True


def main():

    # BLOCK_START
    arr = np.asarray([2, 4, 10, 100, 2.0])
    # print(softmax(arr))
    # print(tf.nn.softmax(arr).numpy())
    assert (np.allclose(tf.nn.softmax(arr).numpy(), softmax(arr)))
    arr = np.asarray([0.0, 0, 0, 1, 0])
    assert (np.allclose(tf.nn.softmax(arr).numpy(), softmax(arr)))
    arr = np.asarray([-750.0, 23, 9, 10, 230])
    assert (np.allclose(tf.nn.softmax(arr).numpy(), softmax(arr)))
    arr = np.ones((4,))
    assert (np.allclose(tf.nn.softmax(arr).numpy(), softmax(arr)))
    arr = np.zeros((4,))
    assert (np.allclose(tf.nn.softmax(arr).numpy(), softmax(arr)))
    # BLOCK_END

    # BLOCK_START
    x = np.asarray([20, 30, -15, 45, 39, -10])
    T = [0.25, 0.75, 1, 1.5, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000]
    x_pos = [i for i, _ in enumerate(x)]

    for idx in range(0, len(T)):
        # plot the result of applying the softmax function
        # with different temperatures on the array x
        softmax_output = softmax(x, T[idx])
        plt.bar(x_pos, softmax_output, color='green')
        plt.title('Temperature t={}'.format(T[idx]))
        # plt.show()  # UNCOMMENT THIS

    # return

    # Q: What happens when we use a large number for the softmax temperature?
    # A: Classes with small values receive a bigger probability compared to the highest valued class
    # Q: What happens when we use a small number (i.e. less than 1) for the softmax temperature?
    # A: The softmax function returns an array in which the highest value class gets the 1 value (so 100%), thus it is
    #    the only visible bar from the barplot
    # Q: In the context of image classification, the predicted class is determined by taking the argmax of the softmax
    #    function. Does the softmax temperature change in any way this prediction?
    # A: No, the temperature does not affect the prediction result since the highest value class still gets the highest
    #    percentage, thus it is still the one chosen by argmax. I have attempted the experiment with t=1000 and still,
    #    the third class is the winner (i.e. has the highest percentage)
    # BLOCK_END

    # BLOCK_START
    from lab2 import cifar10
    cifar_root_dir = 'cifar-10-batches-py'
    _, _, X_test, y_test = cifar10.load_ciaf10(cifar_root_dir)
    indices = np.random.choice(len(X_test), 15)

    display_images, display_labels = X_test[indices], y_test[indices]
    for idx, (img, label) in enumerate(zip(display_images, display_labels)):
        plt.subplot(3, 5, idx + 1)
        plt.imshow(img)
        plt.title(LABELS[label])
        plt.tight_layout()
    plt.show() # UNCOMMENT THIS
    # BLOCK_END

    # BLOCK_START

    cifar_root_dir = 'cifar-10-batches-py'

    # the number of trains performed with different hyper-parameters
    search_iter = 10
    # the batch size
    batch_size = 200
    # number of training steps per training process
    train_steps = 1000

    # load cifar10 dataset
    X_train, y_train, X_test, y_test = cifar10.load_ciaf10(cifar_root_dir)

    # convert the training and test data to floating point
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Reshape the training data such that we have one image per row
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # pre-processing: subtract mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # Bias trick - add 1 to each training example
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    # the search limits for the learning rate and regularization strength
    # we'll use log scale for the search
    lr_bounds = (-7, -2)
    reg_strength_bounds = (-4, -2)

    if not os.path.exists('train'):
        os.mkdir('train')

    best_acc = -1
    best_cls_path = ''

    # Tweaked these values
    learning_rates = [1e-5, 1e-3]
    # learning_rates = [1e-6, 1e-6]
    regularization_strengths = [20657, 20657]  # 80000
    # regularization_strengths = [0.5, 0.5]
    # for each search_iteration, random lr and reg_strength will be chosen

    input_size_flattened = reduce((lambda a, b: a * b), X_train[0].shape)
    results = []
    if TRAIN:
        for idx in range(0, search_iter):
            print('[Search iteration: {} ...]'.format(idx))
            # use log scale for sampling the learning rate
            # lr = pow(10, random.uniform(learning_rates[0], learning_rates[1]))
            lr = random.uniform(learning_rates[0], learning_rates[1])
            reg_strength = random.uniform(regularization_strengths[0], regularization_strengths[1])

            cls = SoftmaxClassifier(input_shape=input_size_flattened, num_classes=cifar10.NUM_CLASSES)
            history = cls.fit(X_train, y_train, lr=lr, reg_strength=reg_strength, verbose=1, \
                              steps=train_steps, bs=batch_size)

            results.append({
                'lr': lr,
                'reg': reg_strength,
                'history': history
            })

            y_train_pred = cls.predict(X_train)
            y_val_pred = cls.predict(X_test)

            train_acc = np.mean(y_train == y_train_pred)

            test_acc = np.mean(y_test == y_val_pred)
            sys.stdout.write(
                '\rlr {:.6f}, reg_strength{:.2f}, test_acc {:.2f}%; train_acc {:.2f}%\n'.format(lr, reg_strength, test_acc * 100,
                                                                                            train_acc * 100))
            cls_path = os.path.join('train', 'softmax_lr{:.4f}_reg{:.4f}-test{:.2f}.npy'.format(lr, reg_strength, test_acc * 100))
            cls.save(cls_path)

            if test_acc > best_acc:
                best_acc = test_acc
                best_cls_path = cls_path

            print('[... end]')

    num_rows = search_iter // 5 + 1
    for idx, res in enumerate(results):
        plt.subplot(num_rows, 5, idx + 1)
        plt.plot(res['history'])
    plt.show()

    # Loading the best classifier
    best_softmax = SoftmaxClassifier(input_shape=input_size_flattened, num_classes=cifar10.NUM_CLASSES)
    best_softmax.load(best_cls_path)

    plt.rcParams['image.cmap'] = 'gray'
    # now let's display the weights for the best model
    weights = best_softmax.get_weights((32, 32, 3))
    w_min = np.amin(weights)
    w_max = np.amax(weights)

    for idx in range(0, cifar10.NUM_CLASSES):
        plt.subplot(2, 5, idx + 1)
        # normalize the weights
        template = 255.0 * (weights[idx, :, :, :].squeeze() - w_min) / (w_max - w_min)
        template = template.astype(np.uint8)
        plt.imshow(template)
        plt.title(cifar10.LABELS[idx])

    plt.show()

    prediction = best_softmax.predict(X_test)
    # TODO your code here
    # use the metrics module to compute the precision, recall and confusion matrix for the best classifier
    print('Accuracy: {}'.format(accuracy_score(y_test, prediction)))
    print('Precision: {}'.format(precision_score(y_test, prediction)))
    print('Recall: {}'.format(recall_score(y_test, prediction)))
    print('[Confusion matrix ...]')
    print(confusion_matrix(y_test, prediction))
    print('[... end]')
    # end TODO your code here
    # BLOCK_END


    # BLOCK_START
    # Sensor quality answers
    # Q1: What can you say about the precision of the measurements that you perform? What about the accuracy of these measurements?
    # A1: If the air quality index stays the same (75) wherever I place the device, my initial guess would be that the
    # device is broken. The underlaying machine learning model might be overfit, that is it is unable to generalize to
    # different situations and always outputs the same value. In this case, the precision of the device is very low
    # since it is quite similar to randomly guessing the same value all over again. The accuracy of the model will be
    # even lower since there are slight chances that the percentage of predictions will be correct.

    # Q2: You determined that the sensor is broken, so you change it with a brand new one. Now everything seems to be ok.
    # To measure the air quality around your house, you place the sensor in different areas: near your favourite scented candle, under your gas central heating exhaust pipe, on your balcony oriented towards the forest/ocean/mountains :) etc.
    # What can you say about the precision of the measurements that you perform? What about the accuracy of these measurements?
    # A2: After fixing the device, the precision of the device will be increased since it will output different values
    # which might be (and hopefuly are) closer to the ground truth. The same for the accuracy. The chances of being accurate
    # increase since the device outputs different measurement values in different locations.
    # BLOCK_END


if __name__ == "__main__":
    main()