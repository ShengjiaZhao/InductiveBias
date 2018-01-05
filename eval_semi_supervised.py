import numpy as np
from sklearn import svm
import time


def semi_supervised_test(train_feature, train_label, test_feature, test_label):
    # Compute pair-wise distance
    x_range = np.sqrt(np.sum(np.square(np.max(train_feature, axis=0) - np.min(train_feature, axis=0))))
    gamma = 0.0001 / x_range
    optimal_gamma = gamma
    optimal_accuracy = 0
    while True:
        classifier = svm.SVC(decision_function_shape='ovr', gamma=gamma)
        classifier.fit(train_feature, train_label)

        pred = classifier.predict(test_feature)
        correct_count = np.sum([1 for j in range(test_feature.shape[0]) if test_label[j] == pred[j]])
        if correct_count > optimal_accuracy:
            optimal_accuracy = correct_count
            optimal_gamma = gamma
        # print("%f %d" % (gamma, correct_count))
        gamma *= 2.0
        if gamma > 100.0:
            break
    optimal_accuracy /= float(test_feature.shape[0])
    return optimal_accuracy, optimal_gamma


def semi_supervised_eval(encoder_x, encoder_z, sess, mnist):
    start_time = time.time()
    print("---------------------> Computing semi-supervised performance")
    train_features = []
    train_labels = []
    for j in range(100):
        bx, by = mnist.train.next_batch(100)
        bx = bx.reshape(-1, 28, 28, 1)
        train_features.append(sess.run(encoder_z, feed_dict={encoder_x: bx}))
        train_labels.append(by)
    train_feature = np.concatenate(train_features, axis=0)
    train_label = np.concatenate(train_labels, axis=0)

    test_features = []
    test_labels = []
    for j in range(100):
        bx, by = mnist.test.next_batch(100)
        bx = bx.reshape(-1, 28, 28, 1)
        test_features.append(sess.run(encoder_z, feed_dict={encoder_x: bx}))
        test_labels.append(by)
    test_feature = np.concatenate(test_features, axis=0)
    test_label = np.concatenate(test_labels)

    accuracy_list = []
    for j in range(10):
        random_ind = np.random.choice(train_feature.shape[0], size=1000, replace=False)
        accuracy, gamma = semi_supervised_test(train_feature[random_ind, :], train_label[random_ind], test_feature,
                                               test_label)
        accuracy_list.append(accuracy)
        print("Processed %d-th batch for 1000 label semi-supervised learning, time elapsed %f" % (
        j, time.time() - start_time))
    accuracy1000 = np.mean(accuracy_list)
    print("Semi-supervised 1000 performance is %f" % accuracy1000)

    accuracy_list = []
    for j in range(10):
        random_ind = np.random.choice(train_feature.shape[0], size=100, replace=False)
        accuracy, gamma = semi_supervised_test(train_feature[random_ind, :], train_label[random_ind], test_feature,
                                               test_label)
        accuracy_list.append(accuracy)
        if j % 10 == 0:
            print("Processed %d-th batch for 100 label semi-supervised learning, time elapsed %f" % (
            j, time.time() - start_time))
    accuracy100 = np.mean(accuracy_list)
    print("Semi-supervised 100 performance is %f" % accuracy100)
    return accuracy1000, accuracy100