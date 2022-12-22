from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
import math

MIN = 0.00000000001 # applied for avoiding log(0) error when computing IG


def load_data():
    """loads data, pre-process it by vectorizing it, then split to training data, validate data
    and test data"""
    result = generate_integrated_data_and_target()
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(result[0])
    Xtrain, Xtest, Ytrain, Ytest = \
        train_test_split(data, result[1], test_size=0.3, train_size=0.7)
    Xval, Xtest, Yval, Ytest = train_test_split(Xtest, Ytest, test_size=0.5, train_size=0.5)
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


def select_model(data):
    """trains decision tree models with two different criteria and five
    different max-depth. then prints result of accuracy for each tree"""
    (Xtrain, Ytrain, Xval, Yval, Xtest, Ytest) = data
    depth = [5, 12, 20, 30, 45]
    criteria = ["entropy", "gini"]
    for deepness in depth:
        for standard in criteria:
            decision_tree = build_model(Xtrain, Ytrain, standard, deepness)
            val_accuracy = validate_model(decision_tree, Xval, Yval)
            print("model: " + standard + "; depth: " + str(deepness) + "; accuracy: " +
                  str(round(val_accuracy, 5)))


def compute_information_gain(data, key_word):
    """computing information gain of given keyword among all sample titles"""
    real_key_branch, fake_key_branch, real_nokey_branch, fake_nokey_branch = \
        simulated_split_result(data, key_word)
    # below will compute information gain with given data above
    h_y = compute_entropy_y(real_key_branch, fake_key_branch,
                            real_nokey_branch, fake_nokey_branch)
    h_y_given_x = compute_entropy_y_given_x(real_key_branch, fake_key_branch,
                                           real_nokey_branch, fake_nokey_branch)
    return h_y - h_y_given_x


def generate_integrated_data_and_target():
    """reads data from given files and create target list corresponding to each
    headline's status: either "fake" or "real".
    """
    fake_file = open("clean_fake.txt", "r")
    real_file = open("clean_real.txt", "r")
    total_train = real_file.readlines()
    real_size = len(total_train)
    fake = fake_file.readlines()
    fake_size = len(fake)
    total_train.extend(fake)
    fake_file.close()
    real_file.close()
    target_list = ["real" for i in range(real_size)]
    target_list.extend(["fake" for j in range(fake_size)])
    return total_train, target_list


def validate_model(trees, Xval, Yval):
    """using validate set for assessing how much validate titles are assigned with
    correct classification as "real" or "fake" """
    accuracy = 0
    for i in range(Xval.shape[0]):
        result = trees.predict(Xval[i])
        if result == Yval[i]:
            accuracy += 1
    return accuracy / Xval.shape[0]


def build_model(Xtrain, Ytrain, criter, depth):
    """helper method for building decision tree models"""
    decision_tree = DecisionTreeClassifier(criterion=criter, max_depth=depth)
    decision_tree.fit(Xtrain, Ytrain)
    return decision_tree


def extract_tree_branch(decision_tree: DecisionTreeClassifier):
    """show the tree graph"""
    tree.plot_tree(decision_tree, max_depth=1)
    plt.show()


def compute_entropy_y(realkey, fakekey, real, fake):
    """helper method for computing entropy before data split"""
    total_sample = real + fake + realkey + fakekey
    real_prob = (realkey + real) / total_sample # all titles which are real divided by total titles
    fake_prob = (fakekey + fake) / total_sample # all fake titles divide by total titles
    entropy_y = real_prob * math.log2(real_prob) + fake_prob * math.log2(fake_prob) # apply entropy formula
    return 0 - entropy_y


def compute_entropy_y_given_x(realkey, fakekey, real, fake):
    """helper method for computing information gain after data splitting"""
    key_total = realkey + fakekey # total # of headlines with keyword
    nokey_total = real + fake # total headlines without the keyword
    real_key_prob, fake_key_prob = realkey / key_total + MIN, fakekey / key_total + MIN # probability for titles with key word
    h_y_key = (key_total / (key_total + nokey_total)) * (
        real_key_prob * math.log2(real_key_prob) + fake_key_prob * math.log2(fake_key_prob)
    ) # apply entropy formula
    real_nokey_prob, fake_nokey_prob = real / nokey_total + MIN, fake / nokey_total + MIN # probability for titles without key word
    h_y_nokey = (nokey_total / (key_total + nokey_total)) * (
        real_nokey_prob * math.log2(real_nokey_prob) + fake_nokey_prob * math.log2(fake_nokey_prob)
    ) # apply entropy formula
    return 0 - (h_y_key + h_y_nokey)


def simulated_split_result(data, key_word):
    """helper method for generating result of splitting with a given keyword,
    then count the number of headlines regarding each type: real and contains keyword, real
    and no keyword, fake with keyword, fake without keyword"""
    Xtrain, Ytrain = data[0], data[1]
    real_key, fake_key, real_nokey, fake_nokey = 0, 0, 0, 0
    for i in range(Xtrain.shape[0]):
        # convert data headlines to strings for search convenience
        str_row = str(Xtrain[i])
        keywords = "(0, {})".format(key_word)
        # count for real headlines how many contains key word
        if Ytrain[i] == "real":
            if keywords in str_row:
                real_key += 1
            else:
                real_nokey += 1
        # count for fake headlines how many contains key word
        elif Ytrain[i] == "fake":
            if keywords in str_row:
                fake_key += 1
            else:
                fake_nokey += 1
    return real_key, fake_key, real_nokey, fake_nokey


if __name__ == "__main__":
    pass
    # vectorized_data = load_data()
#     select_model(vectorized_data)
# # according to validation result, the tree of: model as entropy, depth=30 achieves highest accuracy
#     dec_tree = build_model(vectorized_data[0], vectorized_data[1], "entropy", 30)
#     extract_tree_branch(dec_tree)
#     result5143 = compute_information_gain(vectorized_data, 5143)
#     result5324 = compute_information_gain(vectorized_data, 5324)
#     result1598 = compute_information_gain(vectorized_data, 1598)
#     print("keyword 1598: " + str(result1598) + "\nkeyword 5143: " + str(result5143) +
#           "\nkeyword 5324: " + str(result5324))
