import pandas as pd
import matplotlib.pyplot as plt

#Datasets paths
test_df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/Kfold/full_test_df_results.csv'
full_df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/Kfold/results_full_5day_OCSVM.csv'

#Save path
output_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Plots/Kfold/full_dataset'

#Load in dataset
test_df = pd.read_csv(test_df_path)
full_df = pd.read_csv(full_df_path)


#Filter the dataframes
filtered_test_df = test_df[['Supply_ID', 'Is_Non_Regular', 'anomaly_score']]
filtered_full_df = full_df[['Supply_ID', 'Is_Non_Regular', 'anomaly_score_ocsvm']]


#Separate regular and non regular periods
test_regular_df = filtered_test_df[filtered_test_df['Is_Non_Regular'] == 0]
test_non_regular_df = filtered_test_df[filtered_test_df['Is_Non_Regular'] == 1]
full_regular_df = filtered_full_df[filtered_full_df['Is_Non_Regular'] == 0]
full_non_regular_df = filtered_full_df[filtered_full_df['Is_Non_Regular'] == 1]


#Amounts
non_regular_anomalies = test_non_regular_df[test_non_regular_df['anomaly_score'] < 0.2231]
regular_anomalies = test_regular_df[test_regular_df['anomaly_score'] < 0.2231]

print(len(non_regular_anomalies))
print(len(regular_anomalies))



# Define consistent plot parameters
plot_params = {
    'figsize': (12, 6),
    'title_fontsize': 16,
    'label_fontsize': 14,
    'grid_linestyle': '--',
    'grid_alpha': 0.6,
    'regular_color': 'skyblue',
    'non_regular_color': 'salmon',
    'threshold_color': 'black',
    'threshold_linestyle': '--',
    'threshold_linewidth': 2,
    'marker_size': 50,
    'hist_bins': 30,
    'hist_alpha': 0.7
}

# Helper function to apply consistent styling
def apply_plot_style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=plot_params['title_fontsize'])
    ax.set_xlabel(xlabel, fontsize=plot_params['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=plot_params['label_fontsize'])
    if legend:
        ax.legend(fontsize=12)
    ax.grid(True, linestyle=plot_params['grid_linestyle'], alpha=plot_params['grid_alpha'])

# Plot distributions for test data
plt.figure(figsize=plot_params['figsize'])
plt.subplot(1, 2, 1)
plt.hist(test_regular_df['anomaly_score'], bins=plot_params['hist_bins'], color=plot_params['regular_color'], alpha=plot_params['hist_alpha'], label='Regular')
apply_plot_style(plt.gca(), 'Test Data - Regular Anomaly Scores', 'Anomaly Score', 'Frequency')

plt.subplot(1, 2, 2)
plt.hist(test_non_regular_df['anomaly_score'], bins=plot_params['hist_bins'], color=plot_params['non_regular_color'], alpha=plot_params['hist_alpha'], label='Non-Regular')
apply_plot_style(plt.gca(), 'Test Data - Non-Regular Anomaly Scores', 'Anomaly Score', 'Frequency')

plt.tight_layout()

plt.savefig(f'{output_path}/test_distribution.png')

plt.show()

# Plot distributions for full data
plt.figure(figsize=plot_params['figsize'])
plt.subplot(1, 2, 1)
plt.hist(full_regular_df['anomaly_score_ocsvm'], bins=plot_params['hist_bins'], color=plot_params['regular_color'], alpha=plot_params['hist_alpha'], label='Regular')
apply_plot_style(plt.gca(), 'Full Data - Regular Anomaly Scores', 'Anomaly Score', 'Frequency')

plt.subplot(1, 2, 2)
plt.hist(full_non_regular_df['anomaly_score_ocsvm'], bins=plot_params['hist_bins'], color=plot_params['non_regular_color'], alpha=plot_params['hist_alpha'], label='Non-Regular')
apply_plot_style(plt.gca(), 'Full Data - Non-Regular Anomaly Scores', 'Anomaly Score', 'Frequency')

plt.tight_layout()

plt.savefig(f'{output_path}/full_distribution.png')

plt.show()

# Plot the anomaly scores for test data
plt.figure(figsize=plot_params['figsize'])
plt.scatter(
    test_regular_df.index,
    test_regular_df['anomaly_score'],
    color=plot_params['regular_color'],
    label='Regular',
    alpha=0.7,
    s=plot_params['marker_size']
)
plt.scatter(
    test_non_regular_df.index,
    test_non_regular_df['anomaly_score'],
    color=plot_params['non_regular_color'],
    label='Non-Regular',
    alpha=0.7,
    s=plot_params['marker_size']
)
plt.axhline(y=0.2231, color=plot_params['threshold_color'], linestyle=plot_params['threshold_linestyle'], label='Anomaly Threshold (y=0.2231)', linewidth=plot_params['threshold_linewidth'])
apply_plot_style(plt.gca(), 'Test Data: Anomaly Scores', 'Data Point Index', 'Anomaly Score')

plt.tight_layout()

plt.savefig(f'{output_path}/test_scatter.png')

plt.show()

# Plot the anomaly scores for full data
plt.figure(figsize=plot_params['figsize'])
plt.scatter(
    full_regular_df.index,
    full_regular_df['anomaly_score_ocsvm'],
    color=plot_params['regular_color'],
    label='Regular',
    alpha=0.7,
    s=plot_params['marker_size']
)
plt.scatter(
    full_non_regular_df.index,
    full_non_regular_df['anomaly_score_ocsvm'],
    color=plot_params['non_regular_color'],
    label='Non-Regular',
    alpha=0.7,
    s=plot_params['marker_size']
)
plt.axhline(y=0.2231, color=plot_params['threshold_color'], linestyle=plot_params['threshold_linestyle'], label='Anomaly Threshold (y=0.2231)', linewidth=plot_params['threshold_linewidth'])
apply_plot_style(plt.gca(), 'Full Data: Anomaly Scores', 'Data Point Index', 'Anomaly Score')

plt.tight_layout()

plt.savefig(f'{output_path}/full_scatter.png')

plt.show()