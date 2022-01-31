import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

# get data
df = pd.read_csv('diabetes.csv')

# check nas
df.isnull().sum()
# get stats to check for major imbalances
df.describe()
# get target count. no major imbalance
df.Outcome.value_counts()

# create X and y
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# scale data
X_scl = StandardScaler().fit_transform(X)

# split df
X_train, X_test, y_train, y_test = train_test_split(X_scl, y, test_size=0.2, random_state=10)

# use an unstable classifier like decision tree (with cross validation)
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=10)
scores.mean()

# bagging classifier
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,  # groups of random subsets of df
    max_samples=.8,  # size of training array
    oob_score=True,  # use out of bag(group) sample to estimate error
    random_state=0
)
bag.fit(X_train,y_train)
bag.oob_score_
bag.score(X_test,y_test)
# it has improved

# use cross validation with bagging
scores = cross_val_score(bag, X, y, cv=10)
scores.mean()
# it has improved