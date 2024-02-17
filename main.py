import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pd.read_csv("./data/iris.csv", names=names)


# shape
print("Dataset shape: ", dataset.shape)


# head
print(dataset.head(10))


# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby("class").size())


sns.pairplot(dataset)
plt.show()


dataset.hist()
plt.show()


# Check pandas scatter plot matrix
scatter_matrix(dataset)
plt.show()


array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1
)

# Print or inspect the validation data
print("Validation Features (X_validation):")
print(X_validation)

print("\nValidation Labels (Y_validation):")
print(Y_validation)

# Spot Check Algorithms
models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

    # Compare Algorithms
plt.boxplot(results, labels=names)
plt.title("Algorithm Comparison")
plt.show()


# Make predictions on validation dataset
model = SVC(gamma="auto")
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

for i in range(len(X_validation)):
    # Get the features for the current validation instance
    current_features = X_validation[i]

    # Make prediction for the current validation instance
    prediction = model.predict([current_features])[
        0
    ]  # Predict returns an array, so we take the first element

    # Get the actual label for the current validation instance
    actual_label = Y_validation[i]

    # Print the predicted iris name and the feature values
    print("Predicted Iris:", prediction)
    print("Features:", current_features)
    print("Actual Iris:", actual_label)
    print()


# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
