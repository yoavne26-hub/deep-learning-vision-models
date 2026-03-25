import time
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# ============================================================
# Configuration
# ============================================================

NUM_INPUTS = 784
NUM_CLASSES = 10

BASELINE_HIDDEN_SIZE = 10
HIDDEN_SIZE_CANDIDATES = [10, 64, 128, 256]

DEFAULT_ALPHA = 0.05
DEFAULT_LAMBDA = 0.0
DEFAULT_ITERATIONS = 501
DEFAULT_LOG_EVERY = 100

DATA_SHUFFLE_SEED = 42
INIT_SEED = 42

EFFICIENCY_WEIGHTS = {
    "accuracy": 0.80,
    "runtime": 0.12,
    "params": 0.08,
}


# ============================================================
# Phase 1 - Data loading and cleanup helpers
# ============================================================

def find_training_csv():
    """Locate the MNIST training CSV."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "train.csv",
        Path.home() / "Downloads" / "archive" / "train.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    searched_paths = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not find the MNIST training CSV. Place 'train.csv' next to this script "
        f"or in one of these locations:\n{searched_paths}"
    )


def load_and_split_data(csv_path, shuffle_seed=DATA_SHUFFLE_SEED):
    """
    Load the MNIST CSV and keep the original column-major shape convention:
    X shape = (784, m), Y shape = (m,).
    """
    data = pd.read_csv(csv_path).to_numpy()
    rng = np.random.default_rng(shuffle_seed)
    rng.shuffle(data)

    m, n = data.shape

    data_dev = data[0:1000].T
    Y_dev = data_dev[0].astype(np.int64)
    X_dev = data_dev[1:n] / 255.0

    data_test = data[1000:2000].T
    Y_test = data_test[0].astype(np.int64)
    X_test = data_test[1:n] / 255.0

    data_train = data[2000:m].T
    Y_train = data_train[0].astype(np.int64)
    X_train = data_train[1:n] / 255.0

    return {
        "train": {"X": X_train, "Y": Y_train},
        "dev": {"X": X_dev, "Y": Y_dev},
        "test": {"X": X_test, "Y": Y_test},
    }


# ============================================================
# Phase 1 - Core NumPy neural network math
# ============================================================

def init_params(hidden_size=10, seed=None):
    """Initialize weights for one hidden ReLU layer and a softmax output layer."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((hidden_size, NUM_INPUTS)) * np.sqrt(2.0 / NUM_INPUTS)
    b1 = np.zeros((hidden_size, 1))
    W2 = rng.standard_normal((NUM_CLASSES, hidden_size)) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros((NUM_CLASSES, 1))
    return W1, b1, W2, b2


def relu(Z):
    return np.maximum(Z, 0)


def relu_deriv(Z):
    return Z > 0


def softmax(Z):
    # Shift values for numerical stability.
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y, num_classes=NUM_CLASSES):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y.astype(int), np.arange(Y.size)] = 1
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, one_hot_Y, lambd):
    m = X.shape[1]

    dZ2 = A2 - one_hot_Y
    dW2 = (1.0 / m) * dZ2.dot(A1.T) + (lambd / m) * W2
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * relu_deriv(Z1)
    dW1 = (1.0 / m) * dZ1.dot(X.T) + (lambd / m) * W1
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def get_accuracy(predictions, Y):
    return np.mean(predictions == Y)


def compute_loss(A2, one_hot_Y, eps=1e-12):
    return -np.mean(np.sum(one_hot_Y * np.log(A2 + eps), axis=0))


def evaluate_split(W1, b1, W2, b2, X, Y):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    accuracy = get_accuracy(predictions, Y)
    loss = compute_loss(A2, one_hot(Y))
    return {
        "predictions": predictions,
        "accuracy": float(accuracy),
        "loss": float(loss),
    }


def gradient_descent(
    X,
    Y,
    alpha,
    iterations,
    lambd=0.01,
    hidden_size=10,
    X_dev=None,
    Y_dev=None,
    log_every=10,
    seed=None,
):
    """
    Full-batch gradient descent only.
    The whole training set is used in every iteration.
    """
    W1, b1, W2, b2 = init_params(hidden_size=hidden_size, seed=seed)
    one_hot_Y = one_hot(Y)

    history = {
        "iter": [],
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
    }

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, one_hot_Y, lambd)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % log_every == 0 or i == iterations - 1:
            train_metrics = evaluate_split(W1, b1, W2, b2, X, Y)
            history["iter"].append(i)
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])

            if X_dev is not None and Y_dev is not None:
                dev_metrics = evaluate_split(W1, b1, W2, b2, X_dev, Y_dev)
                history["dev_loss"].append(dev_metrics["loss"])
                history["dev_acc"].append(dev_metrics["accuracy"])
                print(
                    f"Iter {i}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f} | "
                    f"dev_loss={dev_metrics['loss']:.4f}, "
                    f"dev_acc={dev_metrics['accuracy']:.4f}"
                )
            else:
                print(
                    f"Iter {i}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}"
                )

    return W1, b1, W2, b2, history


# ============================================================
# Phase 2 - Baseline experiment pipeline and metrics
# ============================================================

def count_parameters(hidden_size):
    return hidden_size * NUM_INPUTS + hidden_size + NUM_CLASSES * hidden_size + NUM_CLASSES


def estimate_train_work(iterations, num_train_samples, hidden_size):
    return iterations * num_train_samples * (NUM_INPUTS * hidden_size + NUM_CLASSES * hidden_size)


def build_model_dict(W1, b1, W2, b2):
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def run_experiment(
    data_splits,
    hidden_size,
    alpha=DEFAULT_ALPHA,
    lambd=DEFAULT_LAMBDA,
    iterations=DEFAULT_ITERATIONS,
    log_every=DEFAULT_LOG_EVERY,
    seed=INIT_SEED,
):
    X_train = data_splits["train"]["X"]
    Y_train = data_splits["train"]["Y"]
    X_dev = data_splits["dev"]["X"]
    Y_dev = data_splits["dev"]["Y"]
    X_test = data_splits["test"]["X"]
    Y_test = data_splits["test"]["Y"]

    start_time = time.perf_counter()
    W1, b1, W2, b2, history = gradient_descent(
        X_train,
        Y_train,
        alpha=alpha,
        iterations=iterations,
        lambd=lambd,
        hidden_size=hidden_size,
        X_dev=X_dev,
        Y_dev=Y_dev,
        log_every=log_every,
        seed=seed,
    )
    runtime_seconds = time.perf_counter() - start_time

    train_metrics = evaluate_split(W1, b1, W2, b2, X_train, Y_train)
    dev_metrics = evaluate_split(W1, b1, W2, b2, X_dev, Y_dev)
    test_metrics = evaluate_split(W1, b1, W2, b2, X_test, Y_test)

    parameter_count = count_parameters(hidden_size)
    estimated_work = estimate_train_work(iterations, X_train.shape[1], hidden_size)

    return {
        "hidden_size": hidden_size,
        "alpha": alpha,
        "lambda": lambd,
        "iterations": iterations,
        "final_train_accuracy": train_metrics["accuracy"],
        "final_dev_accuracy": dev_metrics["accuracy"],
        "final_test_accuracy": test_metrics["accuracy"],
        "final_train_loss": train_metrics["loss"],
        "final_dev_loss": dev_metrics["loss"],
        "final_test_loss": test_metrics["loss"],
        "runtime_seconds": runtime_seconds,
        "parameter_count": parameter_count,
        "estimated_train_work": estimated_work,
        "history": history,
        "model": build_model_dict(W1, b1, W2, b2),
    }


# ============================================================
# Phase 3 to Phase 5 - Hidden size search and efficiency scoring
# ============================================================

def run_hidden_size_search(
    data_splits,
    hidden_sizes,
    alpha=DEFAULT_ALPHA,
    lambd=DEFAULT_LAMBDA,
    iterations=DEFAULT_ITERATIONS,
    log_every=DEFAULT_LOG_EVERY,
    baseline_result=None,
    seed=INIT_SEED,
):
    results = []

    if baseline_result is not None:
        results.append(baseline_result)

    for hidden_size in hidden_sizes:
        if baseline_result is not None and hidden_size == baseline_result["hidden_size"]:
            continue

        print(f"\n=== Running experiment for hidden_size={hidden_size} ===")
        result = run_experiment(
            data_splits=data_splits,
            hidden_size=hidden_size,
            alpha=alpha,
            lambd=lambd,
            iterations=iterations,
            log_every=log_every,
            seed=seed,
        )
        results.append(result)

    # Dev accuracy is the only model-selection criterion.
    best_by_dev_accuracy = max(results, key=lambda result: result["final_dev_accuracy"])
    return results, best_by_dev_accuracy


def min_max_normalize(values):
    values = np.asarray(values, dtype=float)
    min_value = values.min()
    max_value = values.max()

    if np.isclose(min_value, max_value):
        return np.zeros_like(values)

    return (values - min_value) / (max_value - min_value)


def add_efficiency_scores(results, weights=None):
    if weights is None:
        weights = EFFICIENCY_WEIGHTS

    acc_norm = min_max_normalize([result["final_dev_accuracy"] for result in results])
    time_norm = min_max_normalize([result["runtime_seconds"] for result in results])
    params_norm = min_max_normalize([result["parameter_count"] for result in results])

    for result, acc_value, time_value, params_value in zip(results, acc_norm, time_norm, params_norm):
        efficiency_score = (
            weights["accuracy"] * acc_value
            - weights["runtime"] * time_value
            - weights["params"] * params_value
        )
        result["dev_accuracy_norm"] = float(acc_value)
        result["runtime_norm"] = float(time_value)
        result["parameter_norm"] = float(params_value)
        result["efficiency_score"] = float(efficiency_score)

    return results


def add_log_efficiency_scores(results, runtime_weight=0.4, params_weight=0.6):
    if not results:
        return results

    for result in results:
        denominator = (
            runtime_weight * np.log1p(result["runtime_seconds"])
            + params_weight * np.log1p(result["parameter_count"])
        )

        if np.isclose(denominator, 0.0):
            result["log_efficiency_score"] = 0.0
        else:
            result["log_efficiency_score"] = float(result["final_dev_accuracy"] / denominator)

    return results


def results_to_dataframe(results):
    records = []
    for result in results:
        records.append(
            {
                "hidden_size": result["hidden_size"],
                "alpha": result["alpha"],
                "lambda": result["lambda"],
                "iterations": result["iterations"],
                "train_acc": result["final_train_accuracy"],
                "dev_acc": result["final_dev_accuracy"],
                "test_acc": result["final_test_accuracy"],
                "train_loss": result["final_train_loss"],
                "dev_loss": result["final_dev_loss"],
                "test_loss": result["final_test_loss"],
                "runtime_seconds": result["runtime_seconds"],
                "parameter_count": result["parameter_count"],
                "estimated_train_work": result["estimated_train_work"],
                "dev_accuracy_norm": result.get("dev_accuracy_norm", np.nan),
                "runtime_norm": result.get("runtime_norm", np.nan),
                "parameter_norm": result.get("parameter_norm", np.nan),
                "efficiency_score": result.get("efficiency_score", np.nan),
                "log_efficiency_score": result.get("log_efficiency_score", np.nan),
            }
        )

    return pd.DataFrame(records).sort_values("hidden_size").reset_index(drop=True)


def build_relative_improvement_table(results, baseline_hidden_size=10):
    if not results:
        return pd.DataFrame()

    baseline_result = next(
        (result for result in results if result["hidden_size"] == baseline_hidden_size),
        None,
    )
    if baseline_result is None:
        raise ValueError(f"Could not find baseline result for hidden_size={baseline_hidden_size}.")

    baseline_dev_accuracy = baseline_result["final_dev_accuracy"]
    baseline_test_accuracy = baseline_result["final_test_accuracy"]
    baseline_runtime = baseline_result["runtime_seconds"]
    baseline_params = baseline_result["parameter_count"]
    baseline_efficiency = baseline_result["efficiency_score"]
    baseline_log_efficiency = baseline_result["log_efficiency_score"]

    rows = []
    for result in sorted(results, key=lambda item: item["hidden_size"]):
        dev_accuracy_delta = result["final_dev_accuracy"] - baseline_dev_accuracy
        test_accuracy_delta = result["final_test_accuracy"] - baseline_test_accuracy

        if np.isclose(baseline_dev_accuracy, 0.0):
            dev_accuracy_pct_improvement = 0.0
        else:
            dev_accuracy_pct_improvement = 100.0 * dev_accuracy_delta / baseline_dev_accuracy

        if np.isclose(baseline_test_accuracy, 0.0):
            test_accuracy_pct_improvement = 0.0
        else:
            test_accuracy_pct_improvement = 100.0 * test_accuracy_delta / baseline_test_accuracy

        runtime_ratio = 1.0 if np.isclose(baseline_runtime, 0.0) else result["runtime_seconds"] / baseline_runtime
        parameter_ratio = 1.0 if np.isclose(baseline_params, 0.0) else result["parameter_count"] / baseline_params

        rows.append(
            {
                "hidden_size": result["hidden_size"],
                "final_dev_accuracy": result["final_dev_accuracy"],
                "final_test_accuracy": result["final_test_accuracy"],
                "runtime_seconds": result["runtime_seconds"],
                "parameter_count": result["parameter_count"],
                "efficiency_score": result["efficiency_score"],
                "log_efficiency_score": result["log_efficiency_score"],
                "dev_accuracy_delta": dev_accuracy_delta,
                "test_accuracy_delta": test_accuracy_delta,
                "dev_accuracy_pct_improvement": dev_accuracy_pct_improvement,
                "test_accuracy_pct_improvement": test_accuracy_pct_improvement,
                "runtime_ratio_vs_baseline": runtime_ratio,
                "parameter_ratio_vs_baseline": parameter_ratio,
                "efficiency_delta_vs_baseline": result["efficiency_score"] - baseline_efficiency,
                "log_efficiency_delta_vs_baseline": result["log_efficiency_score"] - baseline_log_efficiency,
            }
        )

    relative_df = pd.DataFrame(rows)
    rounded_columns = {
        "final_dev_accuracy": 4,
        "final_test_accuracy": 4,
        "runtime_seconds": 4,
        "efficiency_score": 4,
        "log_efficiency_score": 4,
        "dev_accuracy_delta": 4,
        "test_accuracy_delta": 4,
        "dev_accuracy_pct_improvement": 2,
        "test_accuracy_pct_improvement": 2,
        "runtime_ratio_vs_baseline": 3,
        "parameter_ratio_vs_baseline": 3,
        "efficiency_delta_vs_baseline": 4,
        "log_efficiency_delta_vs_baseline": 4,
    }
    return relative_df.round(rounded_columns)


# ============================================================
# Phase 6 - Plotting utilities
# ============================================================

def plot_learning_curves(history, title_prefix):
    iterations = history["iter"]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, history["train_acc"], label="Train Accuracy")
    if history["dev_acc"]:
        plt.plot(iterations, history["dev_acc"], label="Dev Accuracy")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy Learning Curve")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, history["train_loss"], label="Train Loss")
    if history["dev_loss"]:
        plt.plot(iterations, history["dev_loss"], label="Dev Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_vs_hidden_size(results_df, column_name, y_label, title):
    plt.figure(figsize=(8, 5))
    plt.plot(results_df["hidden_size"], results_df[column_name], marker="o")
    plt.xlabel("Hidden Size")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_hidden_size_comparison(results_df):
    plot_metric_vs_hidden_size(
        results_df, "dev_acc", "Dev Accuracy", "Dev Accuracy vs Hidden Size"
    )
    plot_metric_vs_hidden_size(
        results_df, "test_acc", "Test Accuracy", "Test Accuracy vs Hidden Size"
    )
    plot_metric_vs_hidden_size(
        results_df, "runtime_seconds", "Runtime (seconds)", "Runtime vs Hidden Size"
    )
    plot_metric_vs_hidden_size(
        results_df, "parameter_count", "Parameter Count", "Parameter Count vs Hidden Size"
    )
    plot_metric_vs_hidden_size(
        results_df, "efficiency_score", "Efficiency Score", "Efficiency Score vs Hidden Size"
    )
    plot_metric_vs_hidden_size(
        results_df,
        "log_efficiency_score",
        "Log Efficiency Score",
        "Log Efficiency Score vs Hidden Size",
    )


def plot_relative_metric_vs_hidden_size(relative_df, column_name, y_label, title):
    plt.figure(figsize=(8, 5))
    plt.plot(relative_df["hidden_size"], relative_df[column_name], marker="o")
    plt.xlabel("Hidden Size")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_relative_improvement_analysis(relative_df, baseline_hidden_size=10):
    plot_relative_metric_vs_hidden_size(
        relative_df,
        "dev_accuracy_delta",
        "Dev Accuracy Delta",
        f"Dev Accuracy Delta vs Baseline ({baseline_hidden_size})",
    )
    plot_relative_metric_vs_hidden_size(
        relative_df,
        "test_accuracy_delta",
        "Test Accuracy Delta",
        f"Test Accuracy Delta vs Baseline ({baseline_hidden_size})",
    )
    plot_relative_metric_vs_hidden_size(
        relative_df,
        "runtime_ratio_vs_baseline",
        "Runtime Ratio vs Baseline",
        f"Runtime Ratio vs Baseline ({baseline_hidden_size})",
    )
    plot_relative_metric_vs_hidden_size(
        relative_df,
        "parameter_ratio_vs_baseline",
        "Parameter Ratio vs Baseline",
        f"Parameter Ratio vs Baseline ({baseline_hidden_size})",
    )

    plt.figure(figsize=(8, 5))
    plt.plot(relative_df["hidden_size"], relative_df["dev_accuracy_delta"], marker="o", label="Dev Accuracy Delta")
    plt.plot(relative_df["hidden_size"], relative_df["test_accuracy_delta"], marker="s", label="Test Accuracy Delta")
    plt.xlabel("Hidden Size")
    plt.ylabel("Accuracy Delta")
    plt.title(f"Accuracy Delta Comparison vs Baseline ({baseline_hidden_size})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overlay_learning_curves(result_a, result_b, label_a, label_b):
    history_a = result_a["history"]
    history_b = result_b["history"]

    plt.figure(figsize=(9, 5))
    plt.plot(history_a["iter"], history_a["train_acc"], label=f"{label_a} Train", linestyle="-")
    plt.plot(history_a["iter"], history_a["dev_acc"], label=f"{label_a} Dev", linestyle="--")
    plt.plot(history_b["iter"], history_b["train_acc"], label=f"{label_b} Train", linestyle="-")
    plt.plot(history_b["iter"], history_b["dev_acc"], label=f"{label_b} Dev", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title(f"Learning Curve Overlay: {label_a} vs {label_b}")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(history_a["iter"], history_a["train_loss"], label=f"{label_a} Train", linestyle="-")
    plt.plot(history_a["iter"], history_a["dev_loss"], label=f"{label_a} Dev", linestyle="--")
    plt.plot(history_b["iter"], history_b["train_loss"], label=f"{label_b} Train", linestyle="-")
    plt.plot(history_b["iter"], history_b["dev_loss"], label=f"{label_b} Dev", linestyle="--")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Overlay: {label_a} vs {label_b}")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# Phase 1 - Prediction visualization compatibility
# ============================================================

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)


def test_prediction(index, W1, b1, W2, b2, X_data, Y_data):
    current_image = X_data[:, index].reshape(-1, 1)
    prediction = make_predictions(current_image, W1, b1, W2, b2)[0]
    label = Y_data[index]

    print("Prediction:", prediction)
    print("Label     :", label)

    image = current_image.reshape(28, 28)
    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def test_prediction_from_result(index, result, X_data, Y_data):
    model = result["model"]
    test_prediction(index, model["W1"], model["b1"], model["W2"], model["b2"], X_data, Y_data)


# ============================================================
# Phase 7 - Reporting helpers
# ============================================================

def print_result_summary(title, result):
    print(f"\n=== {title} ===")
    print(f"hidden_size           : {result['hidden_size']}")
    print(f"alpha                 : {result['alpha']}")
    print(f"lambda                : {result['lambda']}")
    print(f"iterations            : {result['iterations']}")
    print(f"final_train_accuracy  : {result['final_train_accuracy']:.4f}")
    print(f"final_dev_accuracy    : {result['final_dev_accuracy']:.4f}")
    print(f"final_test_accuracy   : {result['final_test_accuracy']:.4f}")
    print(f"final_train_loss      : {result['final_train_loss']:.4f}")
    print(f"final_dev_loss        : {result['final_dev_loss']:.4f}")
    print(f"final_test_loss       : {result['final_test_loss']:.4f}")
    print(f"runtime_seconds       : {result['runtime_seconds']:.4f}")
    print(f"parameter_count       : {result['parameter_count']}")
    print(f"estimated_train_work  : {result['estimated_train_work']}")
    if "efficiency_score" in result:
        print(f"efficiency_score      : {result['efficiency_score']:.4f}")
    if "log_efficiency_score" in result:
        print(f"log_efficiency_score  : {result['log_efficiency_score']:.4f}")


def print_final_report(
    baseline_result,
    best_accuracy_result,
    best_efficiency_result,
    best_log_efficiency_result,
    comparison_df,
    relative_improvement_df,
):
    print_result_summary("Baseline Model", baseline_result)
    print_result_summary("Best Hidden Size by Dev Accuracy", best_accuracy_result)
    print_result_summary("Best Hidden Size by Efficiency Score", best_efficiency_result)
    print_result_summary("Best Hidden Size by Log Efficiency Score", best_log_efficiency_result)

    print("\n=== Full Comparison Table (sorted by hidden_size) ===")
    print(comparison_df.to_string(index=False))

    print("\n=== Comparison Table (sorted by efficiency_score) ===")
    print(comparison_df.sort_values("efficiency_score", ascending=False).to_string(index=False))

    print("\n=== Comparison Table (sorted by log_efficiency_score) ===")
    print(comparison_df.sort_values("log_efficiency_score", ascending=False).to_string(index=False))

    print("\n=== Relative Improvement Table vs Baseline (sorted by hidden_size) ===")
    print(relative_improvement_df.to_string(index=False))

    print("\n=== Relative Improvement Table (sorted by dev_accuracy_delta) ===")
    print(relative_improvement_df.sort_values("dev_accuracy_delta", ascending=False).to_string(index=False))


# ============================================================
# Main script
# ============================================================

def main():
    csv_path = find_training_csv()
    print(f"Loading data from: {csv_path}")
    data_splits = load_and_split_data(csv_path)

    print(f"\n=== Running baseline experiment for hidden_size={BASELINE_HIDDEN_SIZE} ===")
    baseline_result = run_experiment(
        data_splits=data_splits,
        hidden_size=BASELINE_HIDDEN_SIZE,
        alpha=DEFAULT_ALPHA,
        lambd=DEFAULT_LAMBDA,
        iterations=DEFAULT_ITERATIONS,
        log_every=DEFAULT_LOG_EVERY,
        seed=INIT_SEED,
    )

    results, best_accuracy_result = run_hidden_size_search(
        data_splits=data_splits,
        hidden_sizes=HIDDEN_SIZE_CANDIDATES,
        alpha=DEFAULT_ALPHA,
        lambd=DEFAULT_LAMBDA,
        iterations=DEFAULT_ITERATIONS,
        log_every=DEFAULT_LOG_EVERY,
        baseline_result=baseline_result,
        seed=INIT_SEED,
    )

    add_efficiency_scores(results, weights=EFFICIENCY_WEIGHTS)
    add_log_efficiency_scores(results)
    best_efficiency_result = max(results, key=lambda result: result["efficiency_score"])
    best_log_efficiency_result = max(results, key=lambda result: result["log_efficiency_score"])
    comparison_df = results_to_dataframe(results)
    relative_improvement_df = build_relative_improvement_table(
        results, baseline_hidden_size=BASELINE_HIDDEN_SIZE
    )

    print_final_report(
        baseline_result=baseline_result,
        best_accuracy_result=best_accuracy_result,
        best_efficiency_result=best_efficiency_result,
        best_log_efficiency_result=best_log_efficiency_result,
        comparison_df=comparison_df,
        relative_improvement_df=relative_improvement_df,
    )

    plot_learning_curves(baseline_result["history"], title_prefix="Baseline (hidden_size=10)")
    plot_hidden_size_comparison(comparison_df)
    plot_relative_improvement_analysis(
        relative_improvement_df, baseline_hidden_size=BASELINE_HIDDEN_SIZE
    )
    plot_overlay_learning_curves(
        baseline_result,
        best_accuracy_result,
        label_a="Baseline (10)",
        label_b=f"Best Accuracy ({best_accuracy_result['hidden_size']})",
    )
    plot_overlay_learning_curves(
        baseline_result,
        best_efficiency_result,
        label_a="Baseline (10)",
        label_b=f"Best Efficiency ({best_efficiency_result['hidden_size']})",
    )

    # Interpretation:
    # The normalized weighted score is a customizable multi-criteria tradeoff.
    # The log efficiency score is a compact "accuracy per computational cost" style metric.
    # Positive accuracy delta means improvement over the baseline model.
    # Runtime and parameter ratios above 1 mean higher computational cost than the baseline.
    # The best model in pure dev accuracy may be different from the best model in efficiency.
    # The goal of this script is to compare prediction quality against runtime and model cost.

    # Example visualization of individual predictions from the baseline model.
    for index in range(5, 1000, 100):
        test_prediction_from_result(
            index=index,
            result=baseline_result,
            X_data=data_splits["train"]["X"],
            Y_data=data_splits["train"]["Y"],
        )


if __name__ == "__main__":
    main()
