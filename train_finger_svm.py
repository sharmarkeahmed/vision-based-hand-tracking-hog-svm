from joblib import load, dump
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

CLICK_DATA_PATH = "data/hog_finger_click_dataset.joblib"
# Optional: you can also merge with your 11K dataset if you want
USE_11K = False
DATA_11K_PATH = "data/hog_11khands_dataset.joblib"

OUTPUT_MODEL_PATH = "models/hog_finger_svm.joblib"


def main():
    # Load your clicked dataset
    X_click, y_click = load(CLICK_DATA_PATH)
    print("[INFO] Click dataset:", X_click.shape, y_click.shape)

    X_list = [X_click]
    y_list = [y_click]

    if USE_11K:
        X_11k, y_11k = load(DATA_11K_PATH)
        print("[INFO] 11K dataset:", X_11k.shape, y_11k.shape)
        X_list.append(X_11k)
        y_list.append(y_11k)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print("[INFO] Combined dataset:", X.shape, y.shape)
    print("[INFO] Positives:", (y == 1).sum(), "Negatives:", (y == 0).sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("[REPORT]")
    print(classification_report(y_test, y_pred))

    dump(clf, OUTPUT_MODEL_PATH)
    print(f"[SAVED] New SVM model saved to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
