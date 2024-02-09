import os
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

parent_folder = r"D:\\crixodia\\Repositories\\\DQN-DDQN\\\Experiments\\"


def custom_report(y, pred):
    return {
        "Accuracy": accuracy_score(y, pred),
        "Balanced Accuracy": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, average=None),  # "macro"
        "Recall": recall_score(y, pred, average=None),  # "macro"
        "F1": f1_score(y, pred, average=None),  # "macro"
    }


def plot_cf(cf, labels, title=""):
    df_cm = pd.DataFrame(
        cf, index=[i for i in labels.values()], columns=[i for i in labels.values()]
    )

    fig = plt.figure(figsize=(5, 3))

    heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=10
    )
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=10
    )

    plt.ylabel("Actual value")
    plt.xlabel("Predicted value")
    plt.title(title)
    plt.savefig(evalCsvPath.replace(".csv", title + ".png"), bbox_inches="tight")


def plot_accs(accs, baccs, title=""):
    colors = ["skyblue"] * len(accs)
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.barh(
        list(accs.keys()), list(accs.values()), color=colors, label="Accuracy"
    )

    for bar in bars:
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.2f}",
            ha="center",
            va="center",
        )

    plt.plot(
        list(baccs.values()),
        list(baccs.keys()),
        "r--",
        label="Balanced Accuracy",
        linewidth=2,
    )

    plt.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center", ncol=2)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Accuracy")
    plt.title(title)
    plt.savefig(evalCsvPath.replace(".csv", title + ".png"), bbox_inches="tight")


def plot_episode_rewards(dataframe):
    dqn_avg_reward = dataframe["DQNEpisodeReward"].rolling(window=150).mean()
    ddqn_avg_reward = dataframe["DDQNEpisodeReward"].rolling(window=150).mean()

    sns.set(style="whitegrid")
    sns.axes_style("ticks")

    plt.figure(figsize=(6, 3))

    # sns.lineplot(x='EpisodeIndex', y='DQNEpisodeReward', data=dataframe, label='DQN Episode Reward', alpha=0.5)
    sns.lineplot(
        x="EpisodeIndex",
        y=dqn_avg_reward,
        data=dataframe,
        label="DQN Average Reward (Window=150)",
    )

    # sns.lineplot(x='EpisodeIndex', y='DDQNEpisodeReward', data=dataframe, label='DDQN Episode Reward', alpha=0.5)
    sns.lineplot(
        x="EpisodeIndex",
        y=ddqn_avg_reward,
        data=dataframe,
        label="DDQN Average Reward (Window=150)",
    )

    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("DQN and DDQN Average Rewards")

    plt.legend()
    plt.savefig(evalCsvPath.replace(".csv", "rewards.png"), bbox_inches="tight")


experimentData = pd.DataFrame(
    columns=[
        "Experiment",
        "User",
        "Accuracy DQN",
        "Accuracy DDQN",
        "Accuracy PostDQN",
        "Accuracy PostDDQN",
        "Accuracy DQN (noGesture)",
        "Accuracy DDQN (noGesture)",
        "Accuracy PostDQN (noGesture)",
        "Accuracy PostDDQN (noGesture)",
    ]
)

for folder_name in os.listdir(parent_folder):
    folder_path = os.path.join(parent_folder, folder_name)
    if os.path.isdir(folder_path):
        print(folder_name)

        if "u" not in folder_name:
            continue

        user_index = folder_name.find("u")
        user_count = ""
        for i in range(user_index + 1, len(folder_name)):
            if folder_name[i].isdigit():
                user_count += folder_name[i]
            else:
                break

        user_count = int(user_count)

        evalCsvPath = os.path.join(folder_path, folder_name + ".csv")
        df = pd.read_csv(evalCsvPath)

        y = df["ActualGesture"]
        allAccs = dict()
        allbAccs = dict()
        allCfs = dict()

        yDQNpred = df["predDQN"]
        yDDQNpred = df["predDDQN"]

        yPostDQNpred = df["postDQN"]
        yPostDDQNpred = df["postDDQN"]

        ccrDQN = custom_report(y, yDQNpred)
        allAccs["DQN (noGesture)"] = ccrDQN["Accuracy"]
        allbAccs["DQN (noGesture)"] = ccrDQN["Balanced Accuracy"]
        allCfs["DQN (noGesture)"] = confusion_matrix(y, yDQNpred, normalize="true")

        ccrDDQN = custom_report(y, yDDQNpred)
        allAccs["DDQN (noGesture)"] = ccrDDQN["Accuracy"]
        allbAccs["DDQN (noGesture)"] = ccrDDQN["Balanced Accuracy"]
        allCfs["DDQN (noGesture)"] = confusion_matrix(y, yDDQNpred, normalize="true")

        ccrPostDQN = custom_report(y, yPostDQNpred)
        allAccs["PostDQN (noGesture)"] = ccrPostDQN["Accuracy"]
        allbAccs["PostDQN (noGesture)"] = ccrPostDQN["Balanced Accuracy"]
        allCfs["PostDQN (noGesture)"] = confusion_matrix(
            y, yPostDQNpred, normalize="true"
        )

        ccrPostDDQN = custom_report(y, yPostDDQNpred)
        allAccs["PostDDQN (noGesture)"] = ccrPostDDQN["Accuracy"]
        allbAccs["PostDDQN (noGesture)"] = ccrPostDDQN["Balanced Accuracy"]
        allCfs["PostDDQN (noGesture)"] = confusion_matrix(
            y, yPostDDQNpred, normalize="true"
        )

        df = df[~df["ActualGesture"].isin([6])]
        df = df[~df["predDDQN"].isin([6])]
        df = df[~df["predDQN"].isin([6])]
        df = df[~df["postDDQN"].isin([6])]
        df = df[~df["postDQN"].isin([6])]

        yng = df["ActualGesture"]

        yDQNpredng = df["predDQN"]
        yDDQNpredng = df["predDDQN"]

        yPostDQNpredng = df["postDQN"]
        yPostDDQNpredng = df["postDDQN"]

        ccrDQNng = custom_report(yng, yDQNpredng)
        allAccs["DQN"] = ccrDQNng["Accuracy"]
        allbAccs["DQN"] = ccrDQNng["Balanced Accuracy"]
        allCfs["DQN"] = confusion_matrix(yng, yDQNpredng, normalize="true")

        ccrDDQNng = custom_report(yng, yDDQNpredng)
        allAccs["DDQN"] = ccrDDQNng["Accuracy"]
        allbAccs["DDQN"] = ccrDDQNng["Balanced Accuracy"]
        allCfs["DDQN"] = confusion_matrix(yng, yDDQNpredng, normalize="true")

        ccrPostDQNng = custom_report(yng, yPostDQNpredng)
        allAccs["PostDQN"] = ccrPostDQNng["Accuracy"]
        allbAccs["PostDQN"] = ccrPostDQNng["Balanced Accuracy"]
        allCfs["PostDQN"] = confusion_matrix(yng, yPostDQNpredng, normalize="true")

        ccrPostDDQNng = custom_report(yng, yPostDDQNpredng)
        allAccs["PostDDQN"] = ccrPostDDQNng["Accuracy"]
        allbAccs["PostDDQN"] = ccrPostDDQNng["Balanced Accuracy"]
        allCfs["PostDDQN"] = confusion_matrix(yng, yPostDDQNpredng, normalize="true")

        for k, cf in allCfs.items():
            labels = {
                1: "waveIn",
                2: "waveOut",
                3: "fist",
                4: "open",
                5: "pinch",
                6: "noGesture",
            }
            if len(cf) == 5:
                labels = {0: "waveIn", 1: "waveOut", 2: "fist", 3: "open", 4: "pinch"}
            try:
                plot_cf(cf, labels, k)
            except:
                pass

        plot_accs(allAccs, allbAccs, "Accuracy and Balanced Accuracy")

        dataDict = {
            "Experiment": folder_name,
            "User": user_count,
            "Accuracy DQN": ccrDQN["Accuracy"],
            "Accuracy DDQN": ccrDDQN["Accuracy"],
            "Accuracy PostDQN": ccrPostDQN["Accuracy"],
            "Accuracy PostDDQN": ccrPostDDQN["Accuracy"],
            "Accuracy DQN (noGesture)": ccrDQNng["Accuracy"],
            "Accuracy DDQN (noGesture)": ccrDDQNng["Accuracy"],
            "Accuracy PostDQN (noGesture)": ccrPostDQNng["Accuracy"],
            "Accuracy PostDDQN (noGesture)": ccrPostDDQNng["Accuracy"],
        }
        experimentData = pd.concat(
            [experimentData, pd.DataFrame([dataDict], columns=dataDict.keys())]
        )

        if not os.path.isfile(evalCsvPath.replace(".csv", "-tp.csv")):
            continue

        dftp = pd.read_csv(evalCsvPath.replace(".csv", "-tp.csv"))
        plot_episode_rewards(dftp)


experimentData.to_excel(parent_folder + "results.xlsx", index=False)
