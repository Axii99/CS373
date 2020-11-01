import sys
import sklearn
import numpy as np

threshold = 0

def lineprocessor(line):
    line = line.replace('\n','')
    line = line.replace("mother", "1")
    line = line.replace("father", "2")
    line = line.replace("GP", "0")
    line = line.replace("MS", "1")
    line = line.replace("F", "0")
    line = line.replace("M", "1")
    line = line.replace("U", "0")
    line = line.replace("R", "1")
    line = line.replace("U", "0")
    line = line.replace("R", "1")
    line = line.replace("LE3", "0")
    line = line.replace("GT3", "1")
    line = line.replace("T", "0")
    line = line.replace("A", "1")
    line = line.replace("teacher", "1")
    line = line.replace("health", "2")
    line = line.replace("services", "3")
    line = line.replace("at_home", "4")
    line = line.replace("other", "0")
    line = line.replace("home", "1")
    line = line.replace("reputation", "2")
    line = line.replace("course", "3")
    line = line.replace("no", "0")
    line = line.replace("yes", "1")
    line = line.replace("-", "0")
    line = line.replace("+", "1")
    return line


def train():
    train_array =[]
    trainfile = open(sys.argv[1], "r")
    line = trainfile.readline()
    length = 0
    while line:
        line = trainfile.readline()
        length = length + 1
        line = lineprocessor(line)
        temp_array = np.array(line.split(";"))
        train_array = np.concatenate((train_array, temp_array))
    print(length)
    train_array = train_array[:-1]
    train_array = train_array.reshape(length-1, 31)
    class_label = train_array[:, -1]
    train_data = train_array[:, :-1]
    nominal_data = train_data[:, 8:12]
    binary_data = train_data[:, [0, 1, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22]]
    numeric_data = train_data[:, [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29]]
    class_label = np.array(class_label, float)
    train_data = np.array(train_data, float)
    nominal_data = np.array(nominal_data, float)
    binary_data = np.array(binary_data, float)
    numeric_data = np.array(numeric_data, float)
    trainfile.close()
    from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
    b_clf = BernoulliNB()
    g_clf= GaussianNB()
    m_clf = MultinomialNB()
    b_clf.fit(binary_data, class_label)
    g_clf.fit(numeric_data, class_label)
    m_clf.fit(nominal_data, class_label)
    print("--------Train-------------")
    b_predict = b_clf.predict(binary_data)
    b_prob = b_clf.predict_proba(binary_data)
    g_predict = g_clf.predict(numeric_data)
    g_prob = g_clf.predict_proba(numeric_data)
    m_predict = m_clf.predict(nominal_data)
    m_prob = m_clf.predict_proba(nominal_data)
    # print("Accuracy:")
    # print((b_predict != class_label).sum() / (length - 1))
    # print((g_predict != class_label).sum() / (length - 1))
    # print((m_predict != class_label).sum() / (length - 1))
    temp_prob = np.multiply(b_prob, g_prob)
    temp_prob = np.multiply(temp_prob, m_prob)
    prior_square = 0.5 * 0.5
    temp_prob = [prob/(prior_square) for prob in temp_prob]

    result_label = []
    prob_list = []
    for i in range(len(class_label)):
        result = temp_prob[i][1]/temp_prob[i][0]
        if result > 1.0:
            result_label.append(1)

        else:
            result_label.append(0)
        prob_list.append(result)
    prob_list_sorted = prob_list
    prob_list_sorted.sort(reverse=True)
    result_label = np.array(result_label, int)
    # print("final accuracy")
    # print((result_label == class_label).sum() / len(class_label))
    global threshold
    threshold = prob_list_sorted[int(len(prob_list) / 10)]
    print(threshold)

    for i in range(len(class_label)):
        if prob_list[i] > threshold:
            result_label[i] = 1
        else:
            result_label[i] = 0
    print((result_label == 1).sum() / len(class_label))

    # read test file
    print("------------------Begin Test-----------------")
    test_array = []
    testfile = open(sys.argv[2], "r")
    line = testfile.readline()
    length = 0
    while line:
        line = testfile.readline()
        length = length + 1
        line = lineprocessor(line)
        temp_array = np.array(line.split(";"))
        test_array = np.concatenate((test_array, temp_array))
    testfile.close()
    test_array = test_array[:-1]
    test_array = test_array.reshape(length - 1, 31)
    test_label = test_array[:, -1]
    test_data = test_array[:, :-1]
    test_label = np.array(test_label, float)
    test_data = np.array(test_data, float)
    nominal_test = test_data[:, 8:12]
    binary_test = test_data[:, [0, 1, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22]]
    numeric_test = test_data[:, [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29]]
    b_result = test(b_clf, binary_test)
    g_result = test(g_clf, numeric_test)
    m_result = test(m_clf, nominal_test)
    temp_result = np.multiply(b_result, g_result)
    temp_result = np.multiply(temp_result, m_result)
    temp_result = [prob/prior_square for prob in temp_result]
    test_result_label = []
    for i in range(len(temp_result)):
        proba = temp_result[i][1] / temp_result[i][0]
        if proba >= threshold:
            test_result_label.append(1)
        else:
            test_result_label.append(0)
    test_result_label = np.array(test_result_label, float)
    print((test_result_label == 1).sum() / len(test_result_label))
    print((test_result_label == 1).sum())
    print(len(b_result))
    for label in test_result_label:
        if label == 1:
            print("+")
        else:
            print("-")
    return threshold


def test(clf, data_list):
    result_prob = clf.predict_proba(data_list)
    return result_prob



def predict():
    train()
    #test(clf)


predict()
