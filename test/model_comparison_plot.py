import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np

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
    
    # Create a figure with 2 subplots (Trajectory and Error)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # ==========================================
    # Plot 1: Trajectory (Top-Down View X-Z)
    # ==========================================
    ax1 = fig.add_subplot(gs[0])
    
    # Plot Ground Truth
    ax1.plot(df['Truth_X'], df['Truth_Z'], label='Ground Truth', color='black', linewidth=2, alpha=0.6)
    
    # Plot Predictions (Scatter for clarity, or lines)
    # We use scatter for predictions to show where they land relative to the path
    ax1.scatter(df['Pred_Cart_X'], df['Pred_Cart_Z'], label='Cartesian Pred', color='red', s=10, alpha=0.5, marker='x')
    ax1.scatter(df['Pred_Polar_X'], df['Pred_Polar_Z'], label='Polar Pred', color='blue', s=10, alpha=0.5, marker='o')

    # Connect truth to prediction with lines for a few samples to show error vectors? 
    # Maybe too messy. Let's stick to the path.
    
    ax1.set_title('Trajectory (Top-Down View: X-Z)', fontsize=14)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Z Position (Depth) (m)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.axis('equal') # Important for spatial trajectory

    # ==========================================
    # Plot 2: Prediction Error over Time
    # ==========================================
    ax2 = fig.add_subplot(gs[1])

    # Raw Errors
    ax2.plot(df['Frame'], df['Error_Cartesian'], label='Cartesian Error', color='red', alpha=0.3, linewidth=1)
    ax2.plot(df['Frame'], df['Error_Polar'], label='Polar Error', color='blue', alpha=0.3, linewidth=1)

    # Rolling Mean
    window = 10
    df['MA_Cart'] = df['Error_Cartesian'].rolling(window=window).mean()
    df['MA_Polar'] = df['Error_Polar'].rolling(window=window).mean()
    
    ax2.plot(df['Frame'], df['MA_Cart'], color='darkred', linestyle='-', linewidth=2, label=f'Cartesian MA({window})')
    ax2.plot(df['Frame'], df['MA_Polar'], color='darkblue', linestyle='-', linewidth=2, label=f'Polar MA({window})')

    ax2.set_title(f'Prediction Error (Horizon=0.15s)', fontsize=14)
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Euclidean Error (m)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.set_ylim(0, df[['Error_Cartesian', 'Error_Polar']].max().max() * 1.1) # Scale Y axis

    # Add text box with stats
    mean_cart = df['Error_Cartesian'].mean()
    mean_polar = df['Error_Polar'].mean()
    stats_text = (f"Mean Error:\n"
                  f"Cartesian: {mean_cart:.4f} m\n"
                  f"Polar:     {mean_polar:.4f} m\n"
                  f"Improvement: {(mean_cart - mean_polar)/mean_cart*100:.1f}%")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('model_comparison_plot.png')
    print("Plot saved to model_comparison_plot.png")
    plt.show()

if __name__ == "__main__":
    csv_path = "model_comparison_results.csv"
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    plot_comparison(csv_path)
