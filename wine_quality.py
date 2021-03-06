import pandas as pd
from sklearn import tree, metrics
import matplotlib.pyplot as plt

df = pd.read_csv("winequality-red.csv", sep=";")
quality_mapping = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5
}


df.quality = df.quality.map(quality_mapping)

df = df.sample(frac=1).reset_index(drop=True)

df_train = df[:1000]
df_train = df_train.drop("quality", axis="columns")
df_test = df[599:]
df_test = df_test.drop("quality", axis="columns")

train_accuracies= []
test_accuracies= []

for d in range(2,25):
    #print("depth=",d)
    clf = tree.DecisionTreeClassifier(max_depth=d)
    clf.fit(df_train, df[:1000]["quality"])

    train_pred = clf.predict(df_train)
    test_pred = clf.predict(df_test)

    train_accuracy = metrics.accuracy_score(df[:1000]["quality"], train_pred)
    test_accuracy = metrics.accuracy_score(df[599:]["quality"], test_pred)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(10,10))
plt.plot(train_accuracies, label="train accuracy")
plt.plot(test_accuracies, label="test accuracy")
#plt.xticks(range(0, 26, 5))
plt.xlabel("max depth")
plt.ylabel("accuracy")
plt.show()