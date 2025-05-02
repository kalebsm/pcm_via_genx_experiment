import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import math
import os

def plot_capacity_equilibrium_analysis(log_file, output_dir=None):
    """
    Creates three figures from capacity equilibrium log data:
    1. Maximum PMR over iterations
    2. All PMRs over iterations
    3. Individual capacity evolution plots for each generator
    
    Args:
        log_file: Path to the CSV log file
        output_dir: Directory to save output plots (defaults to same directory as log file)
    
    Returns:
        List of paths to the generated figures
    """
    # Read data
    df = pd.read_csv(log_file)
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_palette("colorblind")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(log_file)
    
    # Extract file name without extension for output files
    base_name = os.path.splitext(os.path.basename(log_file))[0]
    
    # Extract generator capacity columns and names
    capacity_columns = [col for col in df.columns if '_capacity_MW' in col]
    generator_names = [col.split('_capacity_MW')[0] for col in capacity_columns]
    
    # Extract PMR columns
    pmr_columns = [gen + "_pmr" for gen in generator_names]
    
    # List to store output file paths
    output_files = []
    
    #-------------------------------------------------------------------------
    # Figure 1: Max PMR over iterations
    #-------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['max_pmr'], marker='o', linestyle='-', 
             linewidth=2, markersize=6, color='darkblue')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Maximum PMR', fontsize=12)
    plt.title('Maximum Profit Margin Ratio (PMR) Over Iterations', fontsize=14)
    plt.grid(True)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add final value annotation
    final_max_pmr = df['max_pmr'].iloc[-1]
    plt.annotate(f'Final: {final_max_pmr:.4f}', 
                 xy=(df['Iteration'].iloc[-1], final_max_pmr),
                 xytext=(5, 5), textcoords='offset points',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
    
    plt.tight_layout()
    
    # Save figure
    max_pmr_file = os.path.join(output_dir, f"{base_name}_max_pmr.png")
    plt.savefig(max_pmr_file, dpi=300, bbox_inches='tight')
    output_files.append(max_pmr_file)
    plt.close()
    
    #-------------------------------------------------------------------------
    # Figure 2: All PMRs over iterations
    #-------------------------------------------------------------------------
    plt.figure(figsize=(12, 8))
    
    # Plot all PMRs together
    for i, (col, name) in enumerate(zip(pmr_columns, generator_names)):
        plt.plot(df['Iteration'], df[col], marker='o', linestyle='-', 
                 label=name, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Profit Margin Ratio (PMR)', fontsize=12)
    plt.title('Generator Profit Margin Ratios', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.axhline(y=0, color='black', linestyle='--')
    
    # # Add max PMR as a separate line
    # plt.plot(df['Iteration'], df['max_pmr'], marker='s', linestyle=':', 
    #          label='Maximum PMR', color='black', linewidth=2)
    
    # Create a table of final values
    last_row = df.iloc[-1]
    textstr = "Final PMRs:\n"
    for i, name in enumerate(generator_names):
        pmr_col = pmr_columns[i]
        textstr += f"{name}: {last_row[pmr_col]:.4f}\n"
    
    # Add text box with final PMRs
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    plt.text(0.02, 0.02, textstr, fontsize=10,
             verticalalignment='bottom', transform=plt.gca().transAxes,
             bbox=props)
    
    plt.tight_layout()
    
    # Save figure
    all_pmrs_file = os.path.join(output_dir, f"{base_name}_all_pmrs.png")
    plt.savefig(all_pmrs_file, dpi=300, bbox_inches='tight')
    output_files.append(all_pmrs_file)
    plt.close()
    
    #-------------------------------------------------------------------------
    # Figure 3: Individual capacity evolution plots
    #-------------------------------------------------------------------------
    # Calculate grid dimensions for subplots
    n_generators = len(generator_names)
    n_cols = min(3, n_generators)
    n_rows = math.ceil(n_generators / n_cols)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_generators > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot capacity evolution for each generator
    for i, (col, name) in enumerate(zip(capacity_columns, generator_names)):
        ax = axes[i]
        ax.plot(df['Iteration'], df[col], marker='o', linestyle='-', 
                linewidth=2, markersize=6, color=sns.color_palette("colorblind")[i % 10])
        
        # Set titles and labels
        ax.set_title(f'{name} Capacity Evolution', fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Capacity (MW)')
        ax.grid(True)
        
        # Add final capacity as text
        last_row = df.iloc[-1]
        final_capacity = last_row[col]
        
        # Add PMR information as a secondary y-axis
        pmr_col = pmr_columns[i]
        ax2 = ax.twinx()
        ax2.plot(df['Iteration'], df[pmr_col], marker='x', linestyle='--', 
                 color='darkred', alpha=0.7)
        ax2.set_ylabel('PMR', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.axhline(y=0, color='darkred', linestyle='--', alpha=0.5)
        
        # Add text with final values
        textstr = f"Final Capacity: {final_capacity:.2f} MW\nFinal PMR: {last_row[pmr_col]:.4f}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    capacity_file = os.path.join(output_dir, f"{base_name}_capacity_evolution.png")
    plt.savefig(capacity_file, dpi=300, bbox_inches='tight')
    output_files.append(capacity_file)
    plt.close()
    
    print(f"Generated {len(output_files)} plot files:")
    for file in output_files:
        print(f" - {file}")
    
    return output_files

if __name__ == "__main__":
    log_file = "/Users/shxryz/Desktop/Research Stuff/spcm_genx_experiment/SPCM/research_systems/2_Hr_BESS/equilibrium_qp_dlac-p.csv"  
    output_dir = "/Users/shxryz/Desktop/Research Stuff/spcm_genx_experiment/figures"
    plot_capacity_equilibrium_analysis(log_file, output_dir)