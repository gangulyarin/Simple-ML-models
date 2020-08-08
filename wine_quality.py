import pandas as pd
from sklearn import tree, metrics

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
df_test = df[:599]
df_test = df_test.drop("quality", axis="columns")

clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(df_train, df[:1000]["quality"])

test_pred = clf.predict(df_test)

train_accuracy = metrics.accuracy_score(df[:599]["quality"], test_pred)
print(train_accuracy)