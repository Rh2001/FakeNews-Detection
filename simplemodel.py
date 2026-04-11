import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class FakeNewsClassifier:
    """Fake News Classification with Baselines + Metadata + Confusion Matrix"""

    def __init__(self, data_path: str = "data/cleaned_data.csv") -> None:
        self.data_path = data_path

        # Data
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        # Logistic Regression PipeLine, clf stands for classifier(logistic regression in this case)
        self.lr_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

        # Naive Bayes PipeLine, clf stands for classifier(naive bayes in this case)
        self.nb_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ])

        self.best_model = None

        # NEW: store predictions (so we don't retrain)
        self.lr_val_pred = None
        self.lr_test_pred = None
        self.nb_val_pred = None
        self.nb_test_pred = None

    # Map labels
    @staticmethod
    def map_label(label):
        label = str(label).lower()

        # Labels that indicate fake news
        if label in ["fake", "rumor", "conspiracy", "junksci"]:
            return 0
        # Labels that indicate real news
        if label in ["reliable", "political"]:
            return 1
        return None

    # Load and split data, with option to include metadata features
    def load_and_split_data(self, use_metadata=False):
        df = pd.read_csv(self.data_path)

        df["binary_label"] = df["type"].apply(self.map_label)
        df = df.dropna(subset=["binary_label"]).copy()

        df["content"] = df["content"].fillna("")

        # Combine content with metadata if requested
        if use_metadata:
            print("\nUsing CONTENT + METADATA features...")
            df["domain"] = df["domain"].fillna("")
            df["title"] = df["title"].fillna("")
            df["authors"] = df["authors"].fillna("")
            df["keywords"] = df["keywords"].fillna("")
            df["source"] = df["source"].fillna("")

            df["text"] = (
                df["domain"] + " " +
                df["content"] + " " +
                df["title"] + " " +
                df["authors"] + " " +
                df["keywords"] + " " +
                df["source"]
            )
        else:
            print("\nUsing CONTENT ONLY...")
            df["text"] = df["content"]

        X = df["text"]
        y = df["binary_label"]

        # 80% train, 20% temp (which will be split into val/test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 50% val, 50% test from temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        self.df = df
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

        print("Train:", len(self.X_train))
        print("Validation:", len(self.X_val))
        print("Test:", len(self.X_test))

    # Baseline models: Logistic Regression and Naive Bayes
    def train_baselines(self):
        print("\n===== BASELINE MODELS =====")

        # Logistic Regression
        print("\nTraining Logistic Regression...")
        self.lr_pipeline.fit(self.X_train, self.y_train)

        self.lr_val_pred = self.lr_pipeline.predict(self.X_val)
        self.lr_test_pred = self.lr_pipeline.predict(self.X_test)

        print("\nLogistic Regression (Validation):")
        self.print_metrics(self.y_val, self.lr_val_pred)

        print("\nLogistic Regression (Test):")
        self.print_metrics(self.y_test, self.lr_test_pred)

        # Naive Bayes
        print("\nTraining Naive Bayes...")
        self.nb_pipeline.fit(self.X_train, self.y_train)

        self.nb_val_pred = self.nb_pipeline.predict(self.X_val)
        self.nb_test_pred = self.nb_pipeline.predict(self.X_test)

        print("\nNaive Bayes (Validation):")
        self.print_metrics(self.y_val, self.nb_val_pred)

        print("\nNaive Bayes (Test):")
        self.print_metrics(self.y_test, self.nb_test_pred)

        # Model comparison
        self.compare_models_on_test(
            self.y_test,
            self.lr_test_pred,
            self.nb_test_pred
        )

        # Confusion matrices
        self.plot_confusion_matrices_side_by_side(
            self.y_test,
            self.lr_test_pred,
            self.nb_test_pred
        )

    # Model comparison table on test set
    def compare_models_on_test(self, y_true, lr_pred, nb_pred):
        print("\n===== MODEL COMPARISON (TEST SET) =====")

        results = pd.DataFrame({
            "Model": ["Logistic Regression", "Naive Bayes"],
            "Accuracy": [
                accuracy_score(y_true, lr_pred),
                accuracy_score(y_true, nb_pred),
            ],
            "Precision": [
                precision_score(y_true, lr_pred),
                precision_score(y_true, nb_pred),
            ],
            "Recall": [
                recall_score(y_true, lr_pred),
                recall_score(y_true, nb_pred),
            ],
            "F1 Score": [
                f1_score(y_true, lr_pred),
                f1_score(y_true, nb_pred),
            ],
        })

        print(results.to_string(index=False))

    # Side-by-side confusion matrix
    def plot_confusion_matrices_side_by_side(self, y_true, lr_pred, nb_pred):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        lr_cm = confusion_matrix(y_true, lr_pred)
        nb_cm = confusion_matrix(y_true, nb_pred)

        ConfusionMatrixDisplay(lr_cm).plot(ax=axes[0], values_format="d")
        axes[0].set_title("Logistic Regression")

        ConfusionMatrixDisplay(nb_cm).plot(ax=axes[1], values_format="d")
        axes[1].set_title("Naive Bayes")

        plt.tight_layout()
        plt.show()

    # Confusion Matrix
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot()
        plt.title(title)
        plt.show()

    # FINAL EVALUATION again for logistic regression, with detailed report and confusion matrix
    def evaluate(self):
        print("\n===== FINAL EVALUATION (LOGISTIC REGRESSION) =====")

        val_pred = self.lr_val_pred
        test_pred = self.lr_test_pred

        print("\nValidation:")
        self.print_metrics(self.y_val, val_pred)

        self.plot_confusion_matrix(
            self.y_val, val_pred, title="Validation Confusion Matrix"
        )

        print("\nTest:")
        self.print_metrics(self.y_test, test_pred)

        self.plot_confusion_matrix(
            self.y_test, test_pred, title="Test Confusion Matrix"
        )

        print("\nDetailed Report:")
        print(classification_report(self.y_test, test_pred))

    @staticmethod
    def print_metrics(y_true, y_pred):
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1 Score:", f1_score(y_true, y_pred))

    # Extracts the top words that are most indicative of fake vs. real news
    def show_top_words(self, top_n=10):
        print("\n===== IMPORTANT WORDS =====")

        feature_names = self.best_model.named_steps["tfidf"].get_feature_names_out()
        coef = self.best_model.named_steps["clf"].coef_[0]

        top_fake = np.argsort(coef)[:top_n]
        top_real = np.argsort(coef)[-top_n:]

        print("\nTop FAKE words:")
        print([feature_names[i] for i in top_fake])

        print("\nTop RELIABLE words:")
        print([feature_names[i] for i in top_real])

    # Run
    def run(self):
        print("\n============================")
        print("TASK 1: CONTENT ONLY")
        print("============================")

        self.load_and_split_data(use_metadata=False)
        self.train_baselines()

        # set best model for interpretability
        self.best_model = self.lr_pipeline

        self.evaluate()
        self.show_top_words()

        print("\n============================")
        print("TASK 2: WITH METADATA")
        print("============================")

        self.load_and_split_data(use_metadata=True)
        self.train_baselines()

        self.best_model = self.lr_pipeline

        self.evaluate()


# Main execution
if __name__ == "__main__":
    classifier = FakeNewsClassifier("data/news_cleaned_2018_02_13_cleaned_20pct.csv")
    classifier.run()