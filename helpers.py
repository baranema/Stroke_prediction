import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from palettable.colorbrewer import qualitative
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix


def addlabels(figure, x, y):
    for i in range(len(x)):
        figure.text(i, y[i] // 2, y[i], ha="center")


def plot_two_kde(df, col, target):
    vals = df[target].value_counts()
    plt.figure(figsize=(15, 8))
    colors = qualitative.Set2_6.hex_colors

    for i in range(0, len(vals)):
        val = vals.keys()[i]
        sns.kdeplot(
            data=df[df[target] == val],
            x=col,
            fill=True,
            common_norm=False,
            color=colors[i],
            alpha=0.5,
            linewidth=0,
            label=f"{target} - {val}",
        )

    plt.legend(loc="upper right")
    plt.title(f"Distribution of {col} and {target}", fontsize=13)
    plt.show()


def plot_distributions(df, missing_col):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    i = 0
    j = 0
    for col in df.columns:
        if col not in ["id", missing_col]:
            if len(list(df[df[missing_col].isnull()][col].value_counts())) > 5:
                axes[i][j].hist(
                    df[df[missing_col].isnull()][col], bins=20, color="#8DA0CB"
                )
                axes[i][j].set_title(f"{col}", fontsize=9)
            else:
                df[df[missing_col].isnull()][col].value_counts().plot(
                    kind="pie",
                    ax=axes[i][j],
                    fontsize=7,
                    autopct="%1.1f%%",
                    colors=qualitative.Set2_4.hex_colors,
                )
                axes[i][j].set_title(f"{col}", fontsize=9)

            j += 1
            if j == 5:
                i += 1
                j = 0

    fig.suptitle(
        f"Different features distribution in entries with missing {missing_col.upper()}",
        fontsize=14,
    )
    plt.show()


def plot_pie_and_bar(df, target):
    f, ax = plt.subplots(1, 2, figsize=(18, 7))
    vals = df[target].value_counts()

    vals.plot.pie(
        autopct="%1.1f%%",
        ax=ax[0],
        colors=qualitative.Set2_4.hex_colors)

    ax[1].bar(list(vals.keys()), list(vals),
              color=qualitative.Set2_4.hex_colors)
    ax[1].grid(False)

    f.suptitle(f"Distribution of {target.upper()}", fontsize=14)
    plt.show()


def plot_pie_and_bar_with_stroke(df, target, stroke_check=True, desc=""):
    f, ax = plt.subplots(1, 2, figsize=(18, 7))
    vals = df[target].value_counts()

    colors = qualitative.Set2_4.hex_colors
    if not stroke_check:
        colors = [colors[0], colors[3]]

    vals.plot.pie(autopct="%1.1f%%", ax=ax[0], colors=colors)

    x = list(vals.keys())
    y = list(vals)

    ax[1].bar(x, y, color=colors)
    ax[1].grid(False)

    addlabels(ax[1], x, y)

    f.suptitle(
        f"Distribution of people {desc} who had or did not have {target.upper()}",
        fontsize=15.5,
    )
    plt.show()

    if stroke_check:
        f, ax = plt.subplots(1, len(vals), figsize=(18, 7))

        for i in range(0, len(vals)):
            val = vals.keys()[i]
            vals_with_stroke = df[df[target] == val]["stroke"].value_counts()
            vals_with_stroke.plot.pie(
                autopct="%1.1f%%",
                ax=ax[i],
                colors=["#2c8559", "#852c3e"],
                wedgeprops=dict(width=0.7, edgecolor="white"),
            )
            ax[i].set_title(f"{target} - {val}")

        f.suptitle(
            f"Distribution of {target.upper()} for people who had or did not have stroke",
            fontsize=14,
        )
        plt.show()


def plot_numerical_histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(df["age"], bins=40, color="#8DA0CB")
    ax2 = axes[0].twinx()
    sns.kdeplot(data=df, x="age", ax=ax2, color="#4c5975")
    ax2.set_xlim((df["age"].min(), df["age"].max()))
    axes[0].set_title(f"Age", fontsize=10)
    ax2.axvline(
        np.mean(df["age"]),
        color="black",
        linestyle="dashed",
        linewidth=1.3,
        label="mean age{:5.0f}".format(np.mean(df["age"])),
    )
    ax2.axvline(
        np.median(df["age"]),
        color="blue",
        linestyle="dashed",
        linewidth=1.3,
        label="median age{:5.0f}".format(np.median(df["age"])),
    )
    ax2.yaxis.set_ticks([])
    ax2.legend(loc=2, prop={"size": 9})

    axes[1].hist(df["bmi"], bins=40, color="#8DA0CB")
    ax2 = axes[1].twinx()
    sns.kdeplot(data=df, x="bmi", ax=ax2, color="#4c5975")
    ax2.set_xlim((df["bmi"].min(), df["bmi"].max()))
    axes[1].set_title(f"BMI", fontsize=10)
    ax2.axvline(
        np.mean(df["bmi"]),
        color="black",
        linestyle="dashed",
        linewidth=1.3,
        label="mean bmi{:5.0f}".format(np.mean(df["bmi"])),
    )
    ax2.axvline(
        np.median(df["bmi"]),
        color="blue",
        linestyle="dashed",
        linewidth=1.3,
        label="median bmi{:5.0f}".format(np.median(df["bmi"])),
    )
    ax2.yaxis.set_ticks([])
    ax2.legend(loc=2, prop={"size": 9})

    axes[2].hist(df["avg_glucose_level"], bins=40, color="#8DA0CB")
    ax2 = axes[2].twinx()
    sns.kdeplot(data=df, x="avg_glucose_level", ax=ax2, color="#4c5975")
    ax2.set_xlim(
        (df["avg_glucose_level"].min(),
         df["avg_glucose_level"].max()))
    axes[2].set_title(f"avg_glucose_level", fontsize=10)
    ax2.axvline(
        np.mean(
            df["avg_glucose_level"]),
        color="black",
        linestyle="dashed",
        linewidth=1.3,
        label="mean avg_glucose_level{:5.0f}".format(
            np.mean(
                df["avg_glucose_level"])),
    )
    ax2.axvline(
        np.median(df["avg_glucose_level"]),
        color="blue",
        linestyle="dashed",
        linewidth=1.3,
        label="median avg_glucose_level{:5.0f}".format(
            np.median(df["avg_glucose_level"])
        ),
    )
    ax2.yaxis.set_ticks([])
    ax2.legend(loc=2, prop={"size": 9})

    fig.suptitle("Distribution of numerical values in the dataset")
    plt.show()


def plot_box_plot_by_col(df, col):
    plt.figure(figsize=(15, 8))
    sns.boxplot(
        x="gender",
        y=col,
        data=df,
        hue="stroke",
        palette=[
            "#852c3e",
            "#2c8559"])

    plt.suptitle(
        f"{col} distribution of people who had and did not have stroke by gender"
    )
    plt.show()


def make_mi_scores(X, y):
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()

    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_classif(
        X, y, discrete_features=discrete_features, random_state=0
    )
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    plt.figure(dpi=100, figsize=(15, 8))
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores, color=qualitative.Set2_6.hex_colors[2])
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()


def plot_scatter_plots(df):
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    sns.regplot(
        x="bmi",
        y="age",
        data=df,
        marker="o",
        color=qualitative.Set2_6.hex_colors[1],
        ax=axes[0],
    )
    sns.regplot(
        x="age",
        y="avg_glucose_level",
        data=df,
        marker="o",
        color=qualitative.Set2_6.hex_colors[1],
        ax=axes[1],
    )
    sns.regplot(
        x="bmi",
        y="avg_glucose_level",
        data=df,
        marker="o",
        color=qualitative.Set2_6.hex_colors[1],
        ax=axes[2],
    )
    fig.suptitle(
        "Scatter plots of continuous features correlations",
        fontsize=27)
    plt.show()


def plot_conf_matrices(results, y):
    f, ax = plt.subplots(1, 4, figsize=(18, 4.5))

    for i in range(0, len(results)):
        model_name = list(results.items())[i][0]
        predictions = list(results.items())[i][1]

        cm = confusion_matrix(y, predictions)
        cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cmn,
            annot=True,
            fmt=".2f",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax[i],
        )

        ax[i].set_title(f"Matrix for {model_name}")
        ax[i].set_xlabel("Predicted labels")
        ax[i].set_ylabel("True labels")

    f.suptitle(
        "Confusion matrices of TEST data for chosen models",
        fontsize=16)
    plt.show()


def plot_roc(X, Y, data_type, models):
    _, ax = plt.subplots(1, 1, figsize=(15, 8))

    model_names = []
    for model in models:
        model_name = type(model.named_steps["model"]).__name__
        model_names.append(model_name)

        y_pred_proba = model.predict_proba(X)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(Y, y_pred_proba)
        plt.plot(fpr, tpr, label=model_name)
        print(f"{model_name} AUC for {data_type} data {metrics.auc(fpr, tpr)}")

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    ax.set_title(f"ROC Curve for {model_names}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="upper right")
    plt.show()


def plot_roc_conf_matrices(
        y1,
        y2,
        model_name,
        data_type1,
        data_type2,
        roc_predictions1,
        roc_predictions2):
    _, ax = plt.subplots(1, 2, figsize=(12, 5.5))

    cm = confusion_matrix(y1, roc_predictions1)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax[0],
    )
    ax[0].set_title(f"{model_name} - Matrix for {data_type1} Data")
    ax[0].set_xlabel("Predicted labels")
    ax[0].set_ylabel("True labels")

    cm = confusion_matrix(y2, roc_predictions2)
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cmn,
        annot=True,
        fmt=".2f",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax[1],
    )
    ax[1].set_title(f"{model_name} - Matrix for {data_type2} Data")
    ax[1].set_xlabel("Predicted labels")
    ax[1].set_ylabel("True labels")

    plt.show()


def plot_numeric_values_box_plots(df):
    plt.rcParams["figure.figsize"] = [15, 8]
    plt.rcParams["figure.autolayout"] = True
    df[["age", "bmi", "avg_glucose_level"]].plot(
        kind="box", title="Numerical features box plots"
    )
    plt.show()


def plot_scatter_for_cols(df, col1, col2, col3):
    sns.relplot(
        x=col1,
        y=col2,
        hue=col3,
        data=df,
        height=8,
        aspect=11.7 / 8,
        palette=["#2eb865", "#bf2644"],
    )
    plt.legend(loc="upper right")
    plt.title(
        f"Scatter plot of {col1} and {col2} for {col3} status",
        fontsize=14)
