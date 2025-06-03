import matplotlib.pyplot as plt
import glob
import os
import re
import json
import yaml
import argparse
import pandas as pd

# -----------------------------------------------------------------------------
flg_loss = False
EXPERIMENTS_FOLDER = "experiments"
# -----------------------------------------------------------------------------

experiments_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), f"../../{EXPERIMENTS_FOLDER}"))
pattern = "experiment_*"
folders = [d for d in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, d)) and re.match(pattern, d)]

# print(folders)
summary_dict = {}
for experiment in folders:
    print(experiment)
    summary_dict[experiment] = {}
    with open(os.path.join(experiments_path, experiment, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        config = argparse.Namespace(**config)
        summary_dict[experiment]['n_sym'] = config.num_symbols
        summary_dict[experiment]['seq_len'] = config.max_length_traces
    for v in ["old", "lrl"]:
        print(v)
        if os.path.exists(os.path.join(experiments_path, experiment,"results", f"metrics_{v}.json")):
            with open(os.path.join(experiments_path, experiment,"results", f"metrics_{v}.json"), "r") as f:
                metrics = json.load(f)
                test_acc = metrics["test_image_classification_accuracy"]
                summary_dict[experiment][f'test_acc_{v}'] = max(test_acc)
                summary_dict[experiment][f'logical_time_{v}'] = sum(metrics["time"]["time_logical"])
        else:
            summary_dict[experiment][f'test_acc_{v}'] = 0.
            summary_dict[experiment][f'logical_time_{v}'] = sum(metrics["time"]["time_logical"])
    # print(summary_dict[experiment][f'test_acc_{v}'])

df = pd.DataFrame.from_dict(summary_dict, orient='index').rename(columns={'test_acc_lrl': 'ILR', 'test_acc_old':'DFA'})
print(df)
# df['ILR_better'] = df['ILR'] > df['DFA']
# print(df['ILR_better'].sum())
# print(df)
# Analisi tempi
df_ = df[df.logical_time_lrl <= df.logical_time_old]
df_g = df_.groupby(['n_sym', 'seq_len'])[['logical_time_old', 'logical_time_lrl']].mean()
# print(df_g)


# print(stop)
# print(df)
grouped = df.groupby(['n_sym', 'seq_len'])[['DFA', 'ILR']].mean()
summary_df = grouped.unstack('seq_len')
summary_df.columns = summary_df.columns.swaplevel(0, 1)
summary_df = summary_df.sort_index(axis=1, level=0)
summary_df = summary_df.round(2)
print(summary_df)

# print(len(summary_dict.keys()))
# for folder in folders:
#     metrics = {}
#     for v in ["old", "lrl"]:
#         try:
#             with open(os.path.join(experiments_path, folder,"results", f"metrics_{v}.json"), "r") as f:
#                 metrics[v] = json.load(f)
#                 print(folder)
#         except:
#             pass
#     try:
#         epochs = list(range(1, len(metrics["lrl"]["loss"])+1))
    
#         fig, ax1 = plt.subplots(figsize=(10, 6))
#         color = 'tab:blue'
#         ax1.set_xlabel('Epochs')
#         ax1.set_ylabel('Accuracy (%)', color=color)
#         l1, = ax1.plot(epochs, metrics["lrl"]["train_image_classification_accuracy"], 'b-', label='LRL Train Accuracy')
#         l2, = ax1.plot(epochs, metrics["lrl"]["test_image_classification_accuracy"], 'b--', label='LRL Test Accuracy')
#         l3, = ax1.plot(epochs, metrics["old"]["train_image_classification_accuracy"], '#FFA500', label='DFA Train Accuracy')
#         l4, = ax1.plot(epochs, metrics["old"]["test_image_classification_accuracy"], '#FFA500', linestyle='dashed', label='DFA Test Accuracy')
#         ticks = list(range(2, len(metrics["lrl"]["train_image_classification_accuracy"])+1, 2))        
#         plt.xticks(ticks)
#         ax1.tick_params(axis='y', labelcolor=color)
        
#         if flg_loss == True:
#             # Create second y-axis for loss
#             ax2 = ax1.twinx()  
#             color = 'tab:red'
#             ax2.set_ylabel('Loss', color=color)
#             l5, = ax2.plot(epochs, metrics["lrl"]["loss"], 'r--', label='LRL Loss')
#             l6, = ax2.plot(epochs, metrics["old"]["loss"], '#FFA500',  label='DFA Loss', linestyle='dotted')
#             ax2.tick_params(axis='y', labelcolor=color)

#         lines = [l1, l2, l3, l4]
#         ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')

#         plt.title(f"Training Metrics: {folder}")
#         fig.tight_layout()

#         # Save and close
#         plt.savefig(os.path.join(experiments_path, folder,"results", f"metrics_compare.png"), dpi=300, bbox_inches='tight')
#         plt.savefig(os.path.join(experiments_path, "_compare_plots", f"metrics_compare_{folder}.png"), dpi=300, bbox_inches='tight')
#         plt.close()
#     except:
#         pass