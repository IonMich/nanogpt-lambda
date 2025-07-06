#!/usr/bin/env python3
"""
NanoGPT Speedrun Results Analyzer
Parses training logs and creates visualizations
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(filename):
    """Parse the NanoGPT training log file"""
    data = {
        'val_steps': [],
        'val_loss': [],
        'val_time_ms': [],
        'train_steps': [],
        'train_time_ms': [],
        'step_avg_ms': []
    }
    
    with open(filename, 'r') as f:
        for line in f:
            # Look for validation results
            # Format: step:1770/1770 val_loss:3.2828 train_time:178912ms step_avg:101.08ms
            val_match = re.search(r'step:(\d+)/\d+ val_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms', line)
            if val_match:
                step, val_loss, train_time, step_avg = val_match.groups()
                data['val_steps'].append(int(step))
                data['val_loss'].append(float(val_loss))
                data['val_time_ms'].append(int(train_time))
                data['step_avg_ms'].append(float(step_avg))
            
            # Also capture training-only steps for finer granularity
            # Format: step:1234/1770 train_time:123456ms step_avg:101.02ms
            train_match = re.search(r'step:(\d+)/\d+ train_time:(\d+)ms step_avg:([\d.]+)ms$', line)
            if train_match and not val_match:  # Only if it's not already a validation line
                step, train_time, step_avg = train_match.groups()
                data['train_steps'].append(int(step))
                data['train_time_ms'].append(int(train_time))
    
    return data

def create_plots(data, filename):
    """Create visualization plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'NanoGPT Speedrun Results - {filename}', fontsize=16, fontweight='bold')
    
    # Convert time to minutes and seconds for easier reading
    val_time_minutes = np.array(data['val_time_ms']) / (1000 * 60)
    val_time_seconds = np.array(data['val_time_ms']) / 1000
    train_time_minutes = np.array(data['train_time_ms']) / (1000 * 60)
    
    # Create interpolated loss curve for smoother visualization
    if len(data['val_steps']) > 1:
        # Interpolate validation loss between checkpoints
        all_steps = list(range(0, max(data['val_steps']) + 1, 10))  # Every 10 steps
        val_loss_interp = np.interp(all_steps, data['val_steps'], data['val_loss'])
        time_interp = np.interp(all_steps, data['val_steps'], val_time_minutes)
    else:
        all_steps = data['val_steps']
        val_loss_interp = data['val_loss']
        time_interp = val_time_minutes
    
    # Plot 1: Validation loss vs steps (fine-grained)
    ax1.plot(all_steps, val_loss_interp, 'b-', label='Validation Loss (interpolated)', linewidth=2, alpha=0.7)
    ax1.scatter(data['val_steps'], data['val_loss'], color='blue', s=30, zorder=5, label='Validation checkpoints')
    ax1.axhline(y=3.28, color='g', linestyle='--', alpha=0.8, linewidth=2, label='Target (≤3.28)')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Progress (Fine-Grained)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=3.0)  # Focus on the relevant range
    
    # Plot 2: Validation loss vs time (THE KEY SPEEDRUN METRIC)
    ax2.plot(time_interp, val_loss_interp, 'b-', linewidth=2, alpha=0.7, label='Validation Loss (interpolated)')
    ax2.scatter(val_time_minutes, data['val_loss'], color='blue', s=40, zorder=5, label='Validation checkpoints')
    ax2.axhline(y=3.28, color='g', linestyle='--', alpha=0.8, linewidth=2, label='Target (≤3.28)')
    ax2.axvline(x=2.992, color='r', linestyle='--', alpha=0.8, linewidth=2, label='Current Record (2.992 min)')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('🏆 SPEEDRUN METRIC: Loss vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=3.0)  # Focus on the relevant range
    
    # Find and highlight when target was achieved
    target_achieved = [(time, loss) for time, loss in zip(val_time_minutes, data['val_loss']) if loss <= 3.28]
    if target_achieved:
        first_target_time, first_target_loss = target_achieved[0]
        ax2.scatter(first_target_time, first_target_loss, color='green', s=150, zorder=10, 
                   marker='*', label=f'Target achieved at {first_target_time:.3f} min')
        ax2.legend()
    
    # Plot 3: Training speed (step average) - using both validation and training data
    if data['step_avg_ms']:
        ax3.plot(data['val_steps'], data['step_avg_ms'], 'orange', linewidth=2, marker='o', markersize=4)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Average Step Time (ms)')
        ax3.set_title('Training Speed per Step')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative training time (fine-grained)
    if data['train_steps'] and data['train_time_ms']:
        # Combine validation and training step data for complete timeline
        all_steps_time = data['val_steps'] + data['train_steps']
        all_times = data['val_time_ms'] + data['train_time_ms']
        
        # Sort by steps
        sorted_data = sorted(zip(all_steps_time, all_times))
        sorted_steps, sorted_times = zip(*sorted_data)
        sorted_times_sec = np.array(sorted_times) / 1000
        
        ax4.plot(sorted_steps, sorted_times_sec, 'purple', linewidth=2, alpha=0.8)
        ax4.scatter(data['val_steps'], val_time_seconds, color='blue', s=30, zorder=5, label='Validation points')
    else:
        ax4.plot(data['val_steps'], val_time_seconds, 'purple', linewidth=2)
    
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Cumulative Time (seconds)')
    ax4.set_title('Total Training Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    return fig

def analyze_results(data, filename):
    """Print analysis of results"""
    print(f"\n🎯 NanoGPT Speedrun Analysis - {filename}")
    print("=" * 60)
    
    if not data['val_loss']:
        print("❌ No training data found in log file")
        return
    
    final_val_loss = data['val_loss'][-1]
    final_time_ms = data['val_time_ms'][-1]
    final_time_sec = final_time_ms / 1000
    final_time_min = final_time_sec / 60
    final_step = data['val_steps'][-1]
    avg_step_time = np.mean(data['step_avg_ms']) if data['step_avg_ms'] else 0
    
    print(f"📊 Final Results:")
    print(f"   • Final validation loss: {final_val_loss:.4f}")
    print(f"   • Training time: {final_time_min:.3f} minutes ({final_time_sec:.1f} seconds)")
    print(f"   • Total steps: {final_step}")
    print(f"   • Average step time: {avg_step_time:.2f}ms")
    
    print(f"\n🎯 Success Criteria:")
    target_achieved = final_val_loss <= 3.28
    print(f"   • Validation loss ≤ 3.28: {'✅ YES' if target_achieved else '❌ NO'} ({final_val_loss:.4f})")
    
    if target_achieved:
        record_time = 2.992  # Current record in minutes
        if final_time_min < record_time:
            print(f"   • 🏆 NEW RECORD! {final_time_min:.3f} min < {record_time} min")
        elif final_time_min < 3.5:
            print(f"   • 🎯 Excellent time! {final_time_min:.3f} min (record: {record_time} min)")
        else:
            print(f"   • ⏱️  Good run: {final_time_min:.3f} min (record: {record_time} min)")
    
    # Find when target was first achieved
    target_indices = [i for i, loss in enumerate(data['val_loss']) if loss <= 3.28]
    if target_indices:
        first_target_idx = target_indices[0]
        target_time_min = data['val_time_ms'][first_target_idx] / (1000 * 60)
        target_step = data['val_steps'][first_target_idx]
        print(f"   • First reached ≤3.28 at: {target_time_min:.3f} minutes (step {target_step})")
        
        if target_time_min < record_time:
            print(f"   • 🎉 This beats the current record by {record_time - target_time_min:.3f} minutes!")
    
    print(f"\n📈 Training Progression:")
    print(f"   • Started at: {data['val_loss'][0]:.4f} validation loss")
    print(f"   • Ended at: {final_val_loss:.4f} validation loss")
    print(f"   • Total improvement: {data['val_loss'][0] - final_val_loss:.4f}")
    
    # Show key checkpoints
    print(f"\n🔍 Key Validation Checkpoints:")
    for i, (step, loss, time_ms) in enumerate(zip(data['val_steps'], data['val_loss'], data['val_time_ms'])):
        time_min = time_ms / (1000 * 60)
        marker = "🎯" if loss <= 3.28 else "  "
        print(f"   {marker} Step {step:4d}: {loss:.4f} at {time_min:.3f} min")
        if i > 0 and i % 5 == 0 and i < len(data['val_steps']) - 1:  # Show every 5th entry, but not the last
            print("   ...")
            break
    
    # Always show the final result
    if len(data['val_steps']) > 1:
        final_time_min = data['val_time_ms'][-1] / (1000 * 60)
        marker = "🎯" if data['val_loss'][-1] <= 3.28 else "  "
        print(f"   {marker} Step {data['val_steps'][-1]:4d}: {data['val_loss'][-1]:.4f} at {final_time_min:.3f} min (FINAL)")

def main():
    # Look for log files in current directory
    log_files = list(Path('.').glob('*.txt'))
    
    if not log_files:
        print("❌ No .txt log files found in current directory")
        print("Make sure you're in the directory with your downloaded log file")
        return
    
    for log_file in log_files:
        print(f"\n📄 Processing: {log_file}")
        
        try:
            data = parse_log_file(log_file)
            
            if not data['val_steps']:
                print(f"❌ No training data found in {log_file}")
                continue
            
            # Create plots
            fig = create_plots(data, log_file.name)
            plot_filename = f"nanogpt_speedrun_results_{log_file.stem}.png"
            
            # Save with high quality
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"📈 High-quality plot saved as: {plot_filename}")
            
            # Show analysis
            analyze_results(data, log_file.name)
            
            # Show the plot
            plt.show()
            
        except Exception as e:
            print(f"❌ Error processing {log_file}: {e}")

if __name__ == "__main__":
    main()
