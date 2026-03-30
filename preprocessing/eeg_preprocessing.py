# =============================================================================
# EEG Sleep Stage Analysis Pipeline
# FAST-NUCES Lahore — Final Year Project 2025-2026
# Non-Invasive Sleep Monitoring Using EEG Signals and Machine Learning
# =============================================================================

import pandas as pd
import numpy as np
import scipy.signal
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: LOAD DATA
# Load raw EEG CSV file recorded via Arduino UNO R4 Minima and
# BioAmp EXG Pill at 512Hz sampling rate with Fp1/Fp2 electrode placement
# =============================================================================
print("Loading EEG data...")
df = pd.read_csv('ahmed_2hr.csv', header=None, names=['timestamp', 'raw_eeg'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['raw_eeg'] = pd.to_numeric(df['raw_eeg'], errors='coerce')
df.dropna(inplace=True)
print(f"Total samples loaded: {len(df)}")
print(f"Recording duration: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])}")
print(df.head())

# =============================================================================
# STEP 2: SIGNAL FILTERING
# Two filters are applied sequentially to isolate clean brain wave signals:
#   1. Notch Filter at 50Hz — removes power line electrical interference
#   2. Bandpass Filter (0.5–30Hz) — retains only relevant brain wave frequencies
#      covering Delta (0.5–3Hz), Theta (4–7Hz), Alpha (8–13Hz), Beta (14–30Hz)
# =============================================================================
print("\nApplying filters...")
sampling_rate = 512
nyquist = 0.5 * sampling_rate

# Notch filter — eliminates 50Hz power line noise from electrical interference
b_notch, a_notch = signal.iirnotch(50.0 / nyquist, Q=0.05, fs=sampling_rate)

# Bandpass filter — isolates brain wave frequency range (0.5Hz to 30Hz)
b_bandpass, a_bandpass = signal.butter(4, [0.5 / nyquist, 30.0 / nyquist], btype='band')

data = df['raw_eeg'].values

# filtfilt applies zero-phase filtering — no time delay distortion in output
filtered = signal.filtfilt(b_notch, a_notch, data)
filtered = signal.filtfilt(b_bandpass, a_bandpass, filtered)
print("Filters applied successfully!")

# =============================================================================
# STEP 3: FEATURE EXTRACTION FUNCTIONS
# Features are extracted from each 30-second epoch using Welch's method
# Welch's method converts the time-domain EEG signal into the frequency domain
# and estimates the Power Spectral Density (PSD) at each frequency band
# =============================================================================

def calculate_psd_features(segment, fs=512):
    """
    Calculates frequency band energy features using Welch's Power Spectral Density.
    
    Each brain wave band corresponds to a specific mental/sleep state:
      - Delta (0.5–3Hz)  : Deep sleep (slow wave sleep)
      - Theta (4–7Hz)    : Light sleep, drowsiness
      - Alpha (8–13Hz)   : Relaxed wakefulness, eyes closed
      - Beta  (14–30Hz)  : Active thinking, alert wakefulness
    
    Parameters:
        segment (np.array): 30-second EEG signal window
        fs (int): Sampling frequency in Hz (default 512Hz)
    
    Returns:
        dict: Energy values for each frequency band + alpha/beta ratio
    """
    f, psd = scipy.signal.welch(segment, fs=fs, nperseg=len(segment))

    alpha_idx = np.where((f >= 8)   & (f <= 13))
    beta_idx  = np.where((f >= 14)  & (f <= 30))
    theta_idx = np.where((f >= 4)   & (f <= 7))
    delta_idx = np.where((f >= 0.5) & (f <= 3))

    E_alpha = np.sum(psd[alpha_idx])
    E_beta  = np.sum(psd[beta_idx])
    E_theta = np.sum(psd[theta_idx])
    E_delta = np.sum(psd[delta_idx])

    # Alpha/Beta ratio — higher ratio indicates relaxed or drowsy state
    alpha_beta_ratio = E_alpha / (E_beta + 1e-10)

    return {
        'E_alpha': E_alpha,
        'E_beta':  E_beta,
        'E_theta': E_theta,
        'E_delta': E_delta,
        'alpha_beta_ratio': alpha_beta_ratio
    }


def calculate_additional_features(segment, fs=512):
    """
    Calculates spectral and time-domain statistical features for ML classification.
    
    Spectral features capture frequency characteristics of the signal.
    Time-domain features capture amplitude and variability of the raw signal.
    
    Parameters:
        segment (np.array): 30-second EEG signal window
        fs (int): Sampling frequency in Hz (default 512Hz)
    
    Returns:
        dict: Spectral and statistical features of the segment
    """
    f, psd = scipy.signal.welch(segment, fs=fs, nperseg=len(segment))

    # Frequency with highest power in this segment
    peak_frequency = f[np.argmax(psd)]

    # Weighted mean frequency — center of mass of the power spectrum
    spectral_centroid = np.sum(f * psd) / (np.sum(psd) + 1e-10)

    # Spectral slope — how power decreases with increasing frequency
    log_f   = np.log(f[1:] + 1e-10)
    log_psd = np.log(psd[1:] + 1e-10)
    spectral_slope = np.polyfit(log_f, log_psd, 1)[0]

    # Time-domain statistical features
    mean_val = np.mean(segment)   # Average signal amplitude
    std_val  = np.std(segment)    # Signal variability
    rms_val  = np.sqrt(np.mean(segment**2))  # Root mean square amplitude

    return {
        'peak_frequency':    peak_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_slope':    spectral_slope,
        'mean':              mean_val,
        'std':               std_val,
        'rms':               rms_val
    }

# =============================================================================
# STEP 4: EPOCH SEGMENTATION AND FEATURE EXTRACTION
# The filtered EEG signal is divided into non-overlapping 30-second epochs
# following the AASM (American Academy of Sleep Medicine) standard for
# sleep stage scoring. Features are extracted from each epoch independently.
# =============================================================================
print("\nExtracting features from 30-second epochs...")
features       = []
timestamps_out = []
window_size    = 30 * 512   # 30 seconds × 512 samples/second = 15360 samples
step_size      = 30 * 512   # Non-overlapping windows (step = window size)

for i in range(0, len(filtered) - window_size, step_size):
    segment   = filtered[i:i + window_size]
    psd_feats = calculate_psd_features(segment, sampling_rate)
    add_feats = calculate_additional_features(segment, sampling_rate)
    combined  = {**psd_feats, **add_feats}
    features.append(combined)
    timestamps_out.append(df['timestamp'].iloc[i])

features_df = pd.DataFrame(features)
features_df.insert(0, 'timestamp', timestamps_out)
print(f"Total 30-second epochs extracted: {len(features_df)}")
print(features_df.head())

# =============================================================================
# STEP 5: SAVE EXTRACTED FEATURES
# Feature matrix saved as CSV for ML model training and future analysis
# =============================================================================
features_df.to_csv('eeg_features.csv', index=False)
print("\nFeatures saved to eeg_features.csv ✅")

# =============================================================================
# STEP 6: ANOMALY DETECTION USING ISOLATION FOREST
# Isolation Forest is an unsupervised ML algorithm that identifies abnormal
# EEG segments without requiring labelled data. It isolates outliers by
# randomly partitioning the feature space — anomalies require fewer splits.
# Contamination parameter set to 0.1 (expects ~10% anomalous epochs)
# =============================================================================
print("\nRunning Anomaly Detection...")
X        = features_df.drop(columns=['timestamp']).values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso_forest = IsolationForest(contamination=0.1, random_state=42)
features_df['anomaly']       = iso_forest.fit_predict(X_scaled)
features_df['anomaly_label'] = features_df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

anomalies = features_df[features_df['anomaly'] == -1]
print(f"Anomalous epochs detected: {len(anomalies)}")
print(f"Normal epochs detected:    {len(features_df) - len(anomalies)}")

# =============================================================================
# STEP 7: RULE-BASED SLEEP STAGE CLASSIFICATION
# Sleep stages are assigned based on the dominant frequency band energy
# in each 30-second epoch, following established EEG sleep scoring criteria:
#   - Delta dominant → Deep Sleep (slow wave sleep, N3)
#   - Theta dominant → Light Sleep or REM
#   - Alpha dominant → Relaxed or Drowsy state
#   - Beta dominant  → Awake and alert
# Note: This serves as heuristic labelling for supervised ML training
# =============================================================================
print("\nClassifying Sleep Stages...")

def classify_sleep_stage(row):
    """
    Assigns a sleep stage label to a 30-second epoch based on dominant
    EEG frequency band energy following AASM scoring guidelines.
    
    Parameters:
        row (pd.Series): Feature row containing band energy values
    
    Returns:
        str: Sleep stage label
    """
    if row['E_delta'] > row['E_alpha'] and row['E_delta'] > row['E_beta']:
        return 'Deep Sleep'
    elif row['E_theta'] > row['E_alpha']:
        return 'Light Sleep / REM'
    elif row['E_alpha'] > row['E_beta']:
        return 'Relaxed / Drowsy'
    else:
        return 'Awake'

features_df['sleep_stage'] = features_df.apply(classify_sleep_stage, axis=1)
print(features_df['sleep_stage'].value_counts())

# =============================================================================
# STEP 8: VISUALIZATION
# Five plots generated to provide comprehensive analysis of the EEG recording:
#   1. Raw EEG signal (first 5 seconds)
#   2. Filtered EEG signal (first 5 seconds)
#   3. Frequency band energy over time
#   4. Sleep stage distribution (pie chart)
#   5. Anomaly detection scatter plot
# =============================================================================
print("\nGenerating visualizations...")
fig = plt.figure(figsize=(20, 24))
gs  = gridspec.GridSpec(4, 2, figure=fig)
fig.suptitle('EEG Sleep Analysis — Non-Invasive Sleep Monitoring System',
             fontsize=16, fontweight='bold')

# Plot 1: Raw EEG Signal
ax1 = fig.add_subplot(gs[0, :])
samples_5sec = 5 * sampling_rate
ax1.plot(df['raw_eeg'].values[:samples_5sec], color='blue', linewidth=0.5)
ax1.set_title('Raw EEG Signal (First 5 Seconds)')
ax1.set_xlabel('Samples')
ax1.set_ylabel('Amplitude')

# Plot 2: Filtered EEG Signal
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(filtered[:samples_5sec], color='green', linewidth=0.5)
ax2.set_title('Filtered EEG Signal After Notch + Bandpass Filter (First 5 Seconds)')
ax2.set_xlabel('Samples')
ax2.set_ylabel('Amplitude')

# Plot 3: Frequency Band Energy Over Time
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(features_df['E_alpha'], label='Alpha — Relaxed (8–13Hz)',    color='green')
ax3.plot(features_df['E_beta'],  label='Beta — Active (14–30Hz)',     color='red')
ax3.plot(features_df['E_theta'], label='Theta — Drowsy (4–7Hz)',      color='orange')
ax3.plot(features_df['E_delta'], label='Delta — Deep Sleep (0.5–3Hz)', color='blue')
ax3.set_title('EEG Frequency Band Energy Over Time')
ax3.set_xlabel('30-Second Epochs')
ax3.set_ylabel('Power Spectral Density')
ax3.legend()

# Plot 4: Sleep Stage Distribution
ax4 = fig.add_subplot(gs[3, 0])
stage_counts = features_df['sleep_stage'].value_counts()
ax4.pie(stage_counts.values, labels=stage_counts.index, autopct='%1.1f%%',
        colors=['navy', 'skyblue', 'lightgreen', 'yellow'])
ax4.set_title('Sleep Stage Distribution')

# Plot 5: Anomaly Detection
ax5 = fig.add_subplot(gs[3, 1])
colors = features_df['anomaly'].map({1: 'blue', -1: 'red'})
ax5.scatter(range(len(features_df)), features_df['E_alpha'],
            c=colors, alpha=0.5, s=5)
ax5.set_title('Isolation Forest Anomaly Detection (Red = Anomalous Epoch)')
ax5.set_xlabel('30-Second Epochs')
ax5.set_ylabel('Alpha Band Energy')

plt.tight_layout()
plt.savefig('eeg_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved to eeg_analysis.png ✅")

# =============================================================================
# STEP 9: SAVE FINAL RESULTS
# Complete feature matrix with sleep stage labels and anomaly flags saved
# =============================================================================
features_df.to_csv('eeg_results_final.csv', index=False)
print("\nFinal results saved to eeg_results_final.csv ✅")

print("\n" + "=" * 50)
print("ANALYSIS SUMMARY")
print("=" * 50)
print(f"Total EEG samples:       {len(df)}")
print(f"Total epochs analyzed:   {len(features_df)}")
print(f"Anomalies detected:      {len(anomalies)}")
print("\nSleep Stage Breakdown:")
print(features_df['sleep_stage'].value_counts())
print("\nOutput files:")
print("  → eeg_features.csv      (extracted feature matrix)")
print("  → eeg_results_final.csv (features + ML labels + anomaly flags)")
print("  → eeg_analysis.png      (visualization dashboard)")
