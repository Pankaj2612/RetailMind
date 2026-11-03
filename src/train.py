from preprocessing import make_model_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


from data_extraction import X,y

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

baseline_pipeline = make_model_pipeline(
    X_train,
    LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42,
    ),
)
