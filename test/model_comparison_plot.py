import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def plot_comparison(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found.")
        return

    if df.empty:
        print("Error: CSV file is empty.")
        return

    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Plot Errors
    plt.plot(df['Frame'], df['Error_Cartesian'], label='Cartesian EKF (Baseline)', color='red', alpha=0.7, linewidth=1.5)
    plt.plot(df['Frame'], df['Error_Polar'], label='Polar EKF (Proposed)', color='blue', alpha=0.8, linewidth=1.5)

    plt.title('Prediction Error Comparison (0.15s Horizon)', fontsize=16)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Prediction Error (m)', fontsize=12)
    plt.legend(fontsize=12)
    
    # Add rolling mean for smoother visualization
    df['MA_Cart'] = df['Error_Cartesian'].rolling(window=10).mean()
    df['MA_Polar'] = df['Error_Polar'].rolling(window=10).mean()
    
    plt.plot(df['Frame'], df['MA_Cart'], color='darkred', linestyle='--', linewidth=2, label='Cartesian MA(10)')
    plt.plot(df['Frame'], df['MA_Polar'], color='darkblue', linestyle='--', linewidth=2, label='Polar MA(10)')

    plt.tight_layout()
    plt.savefig('model_comparison_plot.png')
    print("Plot saved to model_comparison_plot.png")
    plt.show()

if __name__ == "__main__":
    csv_path = "model_comparison_results.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    plot_comparison(csv_path)
