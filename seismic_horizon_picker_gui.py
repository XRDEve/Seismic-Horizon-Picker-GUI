# =============================================================================
# Seismic Horizon Picker GUI - Version 1
# =============================================================================
# Python desktop application for seismic (SBP/SEG-Y) data interpretation
# =============================================================================
# Version 1
# E.Tripoliti 12/06/2025 ucfbetr@ucl.ac.uk
# =============================================================================

# ---------------------------- IMPORTS AND SETUP ------------------------------
import sys
import os
import segyio
import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter1d
import matplotlib
matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for PyQt GUI compatibility
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QMessageBox, QProgressBar,
    QInputDialog, QDialog, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sklearn.cluster import DBSCAN
from collections import defaultdict
import multiprocessing
import pandas as pd

# -------------------------- UTILITY FUNCTIONS -------------------------------
# Signal processing and attribute conversions.

def envelope_to_grayscale(envelope):
    """
    Convert a seismic envelope matrix to an 8-bit grayscale image using
    a robust scaling based on mean ± 3*std (Haralick transform).
    """
    mu = np.mean(envelope)
    delta = np.std(envelope)
    Boundup = mu + 3 * delta
    Boundlow = mu - 3 * delta
    G = (envelope - Boundlow) / (Boundup - Boundlow) * 255
    G = np.clip(G, 0, 255)
    return G.astype(np.uint8)

def twt_to_depth(twt_array, velocity=1500.0):
    """
    Convert two-way travel-time (ms) to depth (m) using acoustic velocity.
    Default velocity is 1500 m/s (typical seawater).
    """
    return (velocity * twt_array / 1000.0) / 2.0

def depth_to_twt(depths, velocity=1500.0): # velocity can change, suggested for marine sediments from 1.4 km/s - 1.8 km/s
    """
    Convert depth (m) to two-way travel-time (ms) using acoustic velocity.
    """
    return (np.array(depths) * 2 * 1000.0) / velocity

def butter_filter(data, fs, kind='low', cutoff=None, order=4, band=None):
    """
    Apply a Butterworth filter (low, high, or band) to a 1D signal.
    - data: 1D input array.
    - fs: Sampling frequency (Hz).
    - kind: 'low', 'high', or 'band'.
    - cutoff/band: cutoff frequency/frequencies in Hz.
    """
    nyq = 0.5 * fs
    if kind == 'low':
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low')
    elif kind == 'high':
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high')
    elif kind == 'band':
        normal_band = [f / nyq for f in band]
        b, a = butter(order, normal_band, btype='band')
    else:
        return data
    y = filtfilt(b, a, data)
    return y

def robust_orientation(trace_indices, sample_indices, window_size=11, min_points=5, interp_gaps=True):
    """
    Compute local dip and azimuth along the main horizon using robust linear regression
    within a moving window. Handles gaps by interpolation, and smooths outliers.
    """
    tr = np.array(trace_indices)
    sm = np.array(sample_indices)
    sort_idx = np.argsort(tr)
    tr = tr[sort_idx]
    sm = sm[sort_idx]
    # Interpolate gaps for continuity
    if interp_gaps and (np.diff(tr).max() > 1):
        all_tr = np.arange(tr[0], tr[-1]+1)
        all_sm = np.interp(all_tr, tr, sm)
        tr, sm = all_tr, all_sm
    # Median filter for outlier suppression
    win = min(window_size, len(sm)//2*2+1)
    sm = median_filter(sm, size=win if win > 2 else 1)
    dip = np.full_like(tr, np.nan, dtype=float)
    azimuth = np.full_like(tr, np.nan, dtype=float)
    half_w = window_size // 2
    # Moving window regression for dip/azimuth
    for i in range(len(tr)):
        lo = max(0, i - half_w)
        hi = min(len(tr), i + half_w + 1)
        t_win = tr[lo:hi]
        s_win = sm[lo:hi]
        if len(t_win) < min_points:
            continue
        A = np.vstack([t_win, np.ones_like(t_win)]).T
        m, c = np.linalg.lstsq(A, s_win, rcond=None)[0]
        dip[i] = np.arctan(m)
        azimuth[i] = (np.degrees(np.arctan2(m, 1)) + 360) % 360
    return tr, sm, dip, azimuth

# ------------------- WORKER THREAD FOR HORIZON PICKING ----------------------
class HorizonWorker(QThread):
    """
    Background thread for automatic horizon picking and DBSCAN clustering.
    Processes seismic data, extracts envelope/phase, finds local/global maxima,
    aligns to phase zero-crossings, clusters picks, and computes orientations.
    """
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, seismic_data, sample_depths, params):
        super().__init__()
        self.seismic_data = seismic_data
        self.sample_depths = sample_depths
        self.params = params

    def run(self):
        try:
            self.progress.emit(5)
            # Analytic signal: envelope and phase extraction
            analytic_signal = hilbert(self.seismic_data, axis=1)
            envelope = np.abs(analytic_signal)
            phase = np.angle(analytic_signal)
            self.progress.emit(15)
            # Gaussian smoothing of envelope for noise reduction
            envelope_smoothed = gaussian_filter(envelope, sigma=self.params['sigma'])
            self.progress.emit(25)
            picks = []
            num_traces, num_samples = phase.shape
            cos_phase = np.cos(phase)
            env_thresh = self.params['env_thresh_ratio'] * np.max(envelope_smoothed)
            # Local maxima detection in envelope + alignment to phase zero-crossings
            for t in range(num_traces):
                maxima = np.argwhere((envelope_smoothed[t][1:-1] > envelope_smoothed[t][:-2]) &
                                    (envelope_smoothed[t][1:-1] > envelope_smoothed[t][2:])).flatten() + 1
                maxima = [m for m in maxima if envelope_smoothed[t, m] > env_thresh and m >= self.params['min_sample'] and m <= self.params['max_sample']]
                maxima = maxima[:self.params['max_horizons']]
                zero_crossings = np.where(np.diff(np.sign(cos_phase[t])))[0]
                for m in maxima:
                    if len(zero_crossings) == 0:
                        continue
                    closest = min(zero_crossings, key=lambda z: abs(z - m))
                    if closest >= self.params['min_sample'] and closest <= self.params['max_sample']:
                        picks.append((t, closest))
            self.progress.emit(40)
            # If no picks ---> empty result
            if not picks:
                self.finished.emit({'picks': [], 'grouped': [], 'orientations': [],
                                    'envelope': envelope_smoothed, 'phase': phase})
                return
            # Cluster picks using DBSCAN (to identify continuous reflectors)
            sample_indices = np.array([[s] for t, s in picks])
            clustering = DBSCAN(eps=self.params['dbscan_eps'], min_samples=self.params['dbscan_min_samples']).fit(sample_indices)
            clustered = defaultdict(list)
            for label, (t, s) in zip(clustering.labels_, picks):
                if label == -1:
                    continue
                clustered[label].append((t, s))
            grouped = list(clustered.values())
            self.progress.emit(60)
            orientations = []
            # Compute local dip and azimuth for each clustered horizon
            for seg in grouped:
                seg = sorted(seg, key=lambda x: x[0])
                trace_indices = [t for t, s in seg]
                sample_indices = [s for t, s in seg]
                tr, sm, dip, azimuth = robust_orientation(
                    trace_indices, sample_indices,
                    window_size=self.params['orientation_window'],
                    min_points=3, interp_gaps=True)
                if len(dip) > 5:
                    dip = median_filter(dip, size=min(7, len(dip)))
                    azimuth = median_filter(azimuth, size=min(7, len(azimuth)))
                seg_orientations = np.stack([dip, azimuth], axis=1)
                orientations.append(seg_orientations)
            self.progress.emit(100)
            # Produce all results for main GUI
            self.finished.emit({
                'picks': picks,
                'grouped': grouped,
                'orientations': orientations,
                'envelope': envelope_smoothed,
                'phase': phase
            })
        except Exception as e:
            self.error.emit(str(e))

# ----------------- DIP/AZIMUTH, REFLECTION COEFFICIENT, WAVEFORM ----------------
class DipAzimuthDialog(QDialog):
    """
    Visualizing dip magnitude, azimuth, and rose diagram
    for the main picked seismic horizon.
    """
    def __init__(self, grouped_horizons, local_orientations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dip, Azimuth, and Rose Diagram")
        layout = QVBoxLayout(self)
        # Dip/Azimuth main plot
        self.fig = Figure(figsize=(12, 9), constrained_layout=True)
        self.ax_dip = self.fig.add_subplot(211)
        self.ax_az = self.fig.add_subplot(212)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # Rose diagram
        self.fig2 = Figure(figsize=(6, 6), constrained_layout=True)
        self.rose_ax = self.fig2.add_subplot(111, polar=True)
        self.canvas2 = FigureCanvas(self.fig2)
        layout.addWidget(self.canvas2)
        # Compute and plot dip, azimuth, rose
        try:
            has_data = bool(grouped_horizons) and bool(local_orientations)
            if has_data:
                # Select main horizon (longest of the cluster, based on continuity)
                main_idx = np.argmax([len(g) for g in grouped_horizons])
                main_horizon = grouped_horizons[main_idx]
                trace_indices = np.array([t for t, s in main_horizon])
                sample_indices = np.array([s for t, s in main_horizon])
                valid = (sample_indices >= 0) & (sample_indices < 1e6) & (~np.isnan(sample_indices))
                trace_indices = trace_indices[valid]
                sample_indices = sample_indices[valid]
                if len(trace_indices) > 2:
                    from scipy.interpolate import interp1d
                    interp_func = interp1d(trace_indices, sample_indices, kind='linear', fill_value="extrapolate")
                    full_traces = np.arange(trace_indices[0], trace_indices[-1]+1)
                    smooth_samples = interp_func(full_traces)
                    smooth_samples = median_filter(smooth_samples, size=15)
                    tr, sm, dip, azimuth = robust_orientation(full_traces, smooth_samples, window_size=21)
                    dip_vals = dip
                    azimuth_vals = azimuth
                    trace_indices_plot = tr
                    # Circular smoothing for azimuth (avoid 0/360 jumps)
                    azimuth_rad = np.deg2rad(azimuth_vals)
                    az_sin = np.sin(azimuth_rad)
                    az_cos = np.cos(azimuth_rad)
                    smooth_az_sin = uniform_filter1d(az_sin, size=21, mode='nearest')
                    smooth_az_cos = uniform_filter1d(az_cos, size=21, mode='nearest')
                    smooth_azimuth_rad = np.arctan2(smooth_az_sin, smooth_az_cos)
                    smooth_azimuth_deg = np.rad2deg(smooth_azimuth_rad) % 360
                else:
                    dip_vals = np.array([])
                    azimuth_vals = np.array([])
                    trace_indices_plot = np.array([])
                    smooth_azimuth_deg = np.array([])
                if len(dip_vals) > 0:
                    # Plot dip magnitude
                    self.ax_dip.plot(trace_indices_plot, dip_vals, color='k', lw=0.75)
                    self.ax_dip.set_title("Dip Magnitude Along Main Horizon", fontsize=14)
                    self.ax_dip.set_ylabel("Dip Magnitude (radians)", fontsize=12)
                    self.ax_dip.tick_params(axis='both', labelsize=10)
                    # Plot azimuth
                    self.ax_az.plot(trace_indices_plot, smooth_azimuth_deg, color='r', lw=0.75, label="Smoothed Azimuth")
                    self.ax_az.set_title("Azimuth Along Main Horizon (Smoothed)", fontsize=14)
                    self.ax_az.set_xlabel("Trace Index", fontsize=12)
                    self.ax_az.set_ylabel("Azimuth (degrees)", fontsize=12)
                    self.ax_az.tick_params(axis='both', labelsize=10)
                    self.ax_az.legend(fontsize=10)
                    # Rose diagram (histogram of azimuth)
                    azimuth_rad_smoothed = np.deg2rad(smooth_azimuth_deg)
                    self.rose_ax.hist(azimuth_rad_smoothed, bins=36, color='r', alpha=0.7)
                    self.rose_ax.set_title('Azimuth Rose Diagram', va='bottom', fontsize=13)
                else:
                    self.ax_dip.set_title("Dip Magnitude Along Main Horizon (No Data)", fontsize=14)
                    self.ax_az.set_title("Azimuth Along Main Horizon (No Data)", fontsize=14)
                    self.rose_ax.set_title('Azimuth Rose Diagram (No Data)', fontsize=13)
            else:
                self.ax_dip.set_title("Dip Magnitude Along Main Horizon (No Data)", fontsize=14)
                self.ax_az.set_title("Azimuth Along Main Horizon (No Data)", fontsize=14)
                self.rose_ax.set_title('Azimuth Rose Diagram (No Data)', fontsize=13)
            self.canvas.draw()
            self.canvas2.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to plot dip/azimuth: {e}")
        # Enable 'Save' buttons for exporting figures
        save_layout = QHBoxLayout()
        self.save_dip_az_btn = QPushButton("Save Dip/Azimuth (PNG, 900 DPI)")
        self.save_rose_btn = QPushButton("Save Rose Diagram (PNG, 900 DPI)")
        self.save_dip_az_btn.clicked.connect(self.save_dip_az_fig)
        self.save_rose_btn.clicked.connect(self.save_rose_fig)
        save_layout.addWidget(self.save_dip_az_btn)
        save_layout.addWidget(self.save_rose_btn)
        layout.addLayout(save_layout)
    def save_dip_az_fig(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Dip/Azimuth Figure", "", "PNG Image (*.png)")
        if fname:
            self.fig.savefig(fname, dpi=900, bbox_inches='tight')
    def save_rose_fig(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Save Rose Diagram", "", "PNG Image (*.png)")
        if fname:
            self.fig2.savefig(fname, dpi=900, bbox_inches='tight')

class ReflectionCoeffDialog(QDialog):
    """
    Displaying and exporting reflection coefficients
    for manual picks on a selected seismic trace.
    """
    def __init__(self, trace, sample_depths, picks, velocity=1500.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reflection Coefficient Analysis")
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        self.trace = trace
        self.sample_depths = sample_depths
        self.picks = picks
        self.velocity = velocity

        # Convert TWT (ms) to depth (m) for pick locations
        depths = twt_to_depth(sample_depths, velocity)
        pick_depths = [depths[idx] for idx in picks]

        # Reference pick selection for normalization
        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference Pick:"))
        self.ref_combo = QComboBox()
        for i, d in enumerate(pick_depths):
            self.ref_combo.addItem(f"{i}: {d:.2f} m")
        ref_layout.addWidget(self.ref_combo)
        layout.addLayout(ref_layout)

        # Main plot setup
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)  # Toolbar goes here
        layout.addWidget(self.canvas)  # Then the canvas

        self.ref_combo.currentIndexChanged.connect(self.replot)
        self.replot()

    def compute_reflection_coefficients(self, trace, picks, window_samples=20, reference_idx=0):
        """
        Compute normalized reflection coefficients for each pick, relative to
        the amplitude at the reference pick.
        """
        coeffs = []
        ref_pick = picks[reference_idx]
        ref_amp = np.max(np.abs(trace[max(0, ref_pick-window_samples//2):ref_pick+window_samples//2+1]))
        for idx in picks:
            win_lo = max(0, idx - window_samples//2)
            win_hi = min(len(trace), idx + window_samples//2 + 1)
            amp = np.max(np.abs(trace[win_lo:win_hi]))
            coeffs.append(amp / ref_amp if ref_amp != 0 else 0)
        return coeffs

    def replot(self):
        """
        Draw (or re-draw) the reflection coefficient plot.
        """
        window_samples = 20
        reference_idx = self.ref_combo.currentIndex()
        coeffs = self.compute_reflection_coefficients(self.trace, self.picks, window_samples, reference_idx)
        depths = twt_to_depth(self.sample_depths, self.velocity)
        pick_depths = [depths[idx] for idx in self.picks]
        self.ax.clear()
        self.ax.plot(coeffs, pick_depths, 'o-', label="Reflection Coefficient")
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Normalized Reflection Coefficient")
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_title("Reflection Coefficient vs Depth")
        self.ax.grid()
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

class WaveformDialog(QDialog):
    """
    Visualizing and manually picking peaks on a single seismic trace.
    Provides filtering tools, manual pick controls, and interactive plot export.
    """
    def __init__(self, trace, sample_depths, trace_idx, parent=None, manual_picks_dict=None, max_sample_index=None):
        super().__init__(parent)
        self.setWindowTitle(f"Waveform at Trace {trace_idx}")
        self.trace = trace
        self.sample_depths = sample_depths
        self.trace_idx = trace_idx
        self.filtered_trace = trace.copy()
        self.fs = self.estimate_sampling_rate()
        self.manual_picks = []
        self.max_sample_index = max_sample_index
        self.xlim = None  # for zooming
        if manual_picks_dict is None:
            self.manual_picks_dict = {}
        else:
            self.manual_picks_dict = manual_picks_dict
        self.initUI()

    def estimate_sampling_rate(self):
        """
        Estimate sampling rate (Hz) from sample depth array.
        """
        diffs = np.diff(self.sample_depths)
        median_dt = np.median(diffs)
        if median_dt <= 0:
            return 1000.0
        return 1000.0 / median_dt

    def initUI(self):
        """
        Build the waveform dialog GUI: filter controls, plot, toolbar, and export buttons.
        """
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["None", "Low-pass", "High-pass", "Band-pass", "Smoothing"])
        controls.addWidget(self.filter_combo)
        # Filter parameter controls
        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(0.01, max(0.01, self.fs / 2000 - 0.01))
        self.cutoff_spin.setSingleStep(0.1)
        self.cutoff_spin.setValue(1.0)
        self.cutoff_spin.setPrefix("Cutoff: ")
        self.cutoff_spin.setSuffix(" kHz")
        controls.addWidget(self.cutoff_spin)
        self.hp_cutoff_spin = QDoubleSpinBox()
        self.hp_cutoff_spin.setRange(0.01, max(0.01, self.fs / 2000 - 0.01))
        self.hp_cutoff_spin.setSingleStep(0.1)
        self.hp_cutoff_spin.setValue(0.5)
        self.hp_cutoff_spin.setPrefix("High: ")
        self.hp_cutoff_spin.setSuffix(" kHz")
        self.hp_cutoff_spin.setVisible(False)
        controls.addWidget(self.hp_cutoff_spin)
        self.lp_cutoff_spin = QDoubleSpinBox()
        self.lp_cutoff_spin.setRange(0.01, max(0.01, self.fs / 2000 - 0.01))
        self.lp_cutoff_spin.setSingleStep(0.1)
        self.lp_cutoff_spin.setValue(2.0)
        self.lp_cutoff_spin.setPrefix("Low: ")
        self.lp_cutoff_spin.setSuffix(" kHz")
        self.lp_cutoff_spin.setVisible(False)
        controls.addWidget(self.lp_cutoff_spin)
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(1, 101)
        self.smooth_spin.setValue(5)
        self.smooth_spin.setPrefix("Window: ")
        self.smooth_spin.setSuffix(" samples")
        self.smooth_spin.setVisible(False)
        controls.addWidget(self.smooth_spin)
        self.apply_btn = QPushButton("Apply")
        controls.addWidget(self.apply_btn)
        self.save_btn = QPushButton("Save Waveform (PNG, 900 DPI)")
        self.save_btn.clicked.connect(self.save_fig)
        controls.addWidget(self.save_btn)
        self.clear_picks_btn = QPushButton("Clear Manual Picks")
        self.clear_picks_btn.clicked.connect(self.clear_manual_picks)
        controls.addWidget(self.clear_picks_btn)
        # Restore Show Manual Picks and Export/Reflection Buttons if desired
        self.show_picks_btn = QPushButton("Show Manual Picks on SBP")
        self.show_picks_btn.clicked.connect(self.handle_show_manual_picks)
        controls.addWidget(self.show_picks_btn)
        self.export_picks_btn = QPushButton("Export Manual Picks (.xlsx)")
        self.export_picks_btn.clicked.connect(self.handle_export_manual_picks)
        controls.addWidget(self.export_picks_btn)
        self.reflection_btn = QPushButton("Reflection Coefficient")
        self.reflection_btn.clicked.connect(self.handle_reflection_coeff)
        controls.addWidget(self.reflection_btn)
        layout.addLayout(controls)
        # Matplotlib figure and toolbar for waveform visualization
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # --- Connect filter controls and apply logic ---
        self.filter_combo.currentIndexChanged.connect(self.update_controls_visibility)
        self.filter_combo.currentIndexChanged.connect(self.apply_filter_and_plot)
        self.cutoff_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.hp_cutoff_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.lp_cutoff_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.smooth_spin.valueChanged.connect(self.apply_filter_and_plot)
        self.apply_btn.clicked.connect(self.apply_filter_and_plot)
        self.update_controls_visibility()
        # Connect picking
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.plot_waveform(self.filtered_trace, label="Raw Signal", color='gray')

    def update_controls_visibility(self):
        """
        Show/hide filter parameter controls depending on selected filter type.
        """
        ftype = self.filter_combo.currentText()
        self.cutoff_spin.setVisible(ftype in ["Low-pass", "High-pass"])
        self.hp_cutoff_spin.setVisible(ftype == "Band-pass")
        self.lp_cutoff_spin.setVisible(ftype == "Band-pass")
        self.smooth_spin.setVisible(ftype == "Smoothing")

    def apply_filter_and_plot(self):
        """
        Apply the selected filter to the trace and update the plot.
        """
        ftype = self.filter_combo.currentText()
        y = self.trace
        try:
            if ftype == "None":
                self.filtered_trace = y.copy()
                self.plot_waveform(self.filtered_trace, label="Raw Signal", color='gray')
            elif ftype == "Low-pass":
                cutoff = self.cutoff_spin.value() * 1000
                self.filtered_trace = butter_filter(y, self.fs, 'low', cutoff)
                self.plot_waveform(self.filtered_trace, label=f"Low-pass {self.cutoff_spin.value():.2f}kHz", color='b')
            elif ftype == "High-pass":
                cutoff = self.cutoff_spin.value() * 1000
                self.filtered_trace = butter_filter(y, self.fs, 'high', cutoff)
                self.plot_waveform(self.filtered_trace, label=f"High-pass {self.cutoff_spin.value():.2f}kHz", color='g')
            elif ftype == "Band-pass":
                hp = self.hp_cutoff_spin.value() * 1000
                lp = self.lp_cutoff_spin.value() * 1000
                if hp >= lp:
                    QMessageBox.warning(self, "Invalid band", "High cutoff must be lower than low cutoff")
                    return
                self.filtered_trace = butter_filter(y, self.fs, 'band', band=[hp, lp])
                self.plot_waveform(self.filtered_trace, label=f"Band {self.hp_cutoff_spin.value():.2f}-{self.lp_cutoff_spin.value():.2f}kHz", color='m')
            elif ftype == "Smoothing":
                window = self.smooth_spin.value()
                self.filtered_trace = np.convolve(y, np.ones(window)/window, mode='same')
                self.plot_waveform(self.filtered_trace, label=f"Smoothing {window}", color='r')
        except Exception as e:
            QMessageBox.critical(self, "Filter Error", f"Error applying filter: {e}")

    def plot_waveform(self, y, label="Raw Signal", color='gray'):
        """
        Plot the current waveform, manual picks, and cutoff line (if set).
        """
        x_depth = twt_to_depth(self.sample_depths)
        self.ax.clear()
        if label != "Raw Signal":
            self.ax.plot(x_depth, self.trace, color='gray', lw=0.5, label="Raw Signal", zorder=1)
        self.ax.plot(x_depth, y, color=color, lw=0.5, label=label, zorder=2)
        if self.max_sample_index is not None:
            cutoff_depth = twt_to_depth(self.sample_depths[self.max_sample_index])
            self.ax.axvline(cutoff_depth, color='red', linestyle='--', lw=1.5, label="Picking Cutoff")
        if len(self.manual_picks) > 0:
            for i, idx in enumerate(self.manual_picks):
                self.ax.axvline(x_depth[idx], color='g', linestyle='-', lw=1.5, alpha=0.85, label="Manual Pick" if i == 0 else "")
        self.ax.set_xlabel("Depth (m)")
        self.ax.set_ylabel("Amplitude")
        self.ax.set_title(f"Waveform at Selected Trace {self.trace_idx}")
        handles, labels = self.ax.get_legend_handles_labels()
        uniq = list(dict(zip(labels, handles)).items())
        if uniq:
            self.ax.legend([h for l, h in uniq], [l for l, h in uniq])
        if self.xlim is not None:
            self.ax.set_xlim(self.xlim)
        self.fig.tight_layout()
        self.canvas.draw()

    def on_canvas_click(self, event):
        """
        Handle manual pick (left-click) on the waveform plot.
        """
        if event.inaxes != self.ax or event.button != 1:
            return
        x_depth = twt_to_depth(self.sample_depths)
        click_depth = event.xdata
        if click_depth is None:
            return
        idx = np.argmin(np.abs(x_depth - click_depth))
        if self.max_sample_index is not None and idx > self.max_sample_index:
            QMessageBox.warning(self, "Invalid Pick", "Cannot pick below the cutoff depth.")
            return
        if idx not in self.manual_picks:
            self.manual_picks.append(idx)
            self.plot_waveform(self.filtered_trace,
                               label=self.filter_combo.currentText() if self.filter_combo.currentText() != "None" else "Raw Signal",
                               color={'None':'gray','Low-pass':'b','High-pass':'g','Band-pass':'m','Smoothing':'r'}.get(self.filter_combo.currentText(), 'gray'))
            self.manual_picks_dict[self.trace_idx] = list(self.manual_picks)

    def clear_manual_picks(self):
        """
        Clear all manual picks for this trace and update the plot.
        """
        self.manual_picks = []
        self.plot_waveform(self.filtered_trace,
                           label=self.filter_combo.currentText() if self.filter_combo.currentText() != "None" else "Raw Signal",
                           color={'None':'gray','Low-pass':'b','High-pass':'g','Band-pass':'m','Smoothing':'r'}.get(self.filter_combo.currentText(), 'gray'))
        if self.trace_idx in self.manual_picks_dict:
            del self.manual_picks_dict[self.trace_idx]

    def save_fig(self):
        """
        Export the current waveform plot as a PNG image.
        """
        fname, _ = QFileDialog.getSaveFileName(self, "Save Waveform Figure", "", "PNG Image (*.png)")
        if fname:
            self.fig.savefig(fname, dpi=900, bbox_inches='tight')

    def handle_export_manual_picks(self):
        """
        Export manual picks for this trace to an Excel file.
        """
        if self.trace_idx not in self.manual_picks_dict or not self.manual_picks_dict[self.trace_idx]:
            QMessageBox.information(self, "No Picks", "No manual picks to export for this trace.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export Manual Picks", "", "Excel Files (*.xlsx)")
        if not fname:
            return
        rows = []
        for idx in self.manual_picks_dict[self.trace_idx]:
            depth = twt_to_depth(self.sample_depths[idx])
            rows.append({'Trace Index': self.trace_idx, 'Sample Index': idx, 'Depth (m)': depth})
        df = pd.DataFrame(rows)
        df.to_excel(fname, index=False)
        QMessageBox.information(self, "Export Complete", f"Manual picks exported to {fname}")

    def handle_show_manual_picks(self):
        """
        Request the main window to plot manual picks for all traces.
        """
        if self.parent() is not None and hasattr(self.parent(), "plot_manual_picks_on_section"):
            self.parent().plot_manual_picks_on_section()

    def handle_reflection_coeff(self):
        """
        Show the Reflection Coefficient dialog for current manual picks.
        """
        if self.trace_idx not in self.manual_picks_dict or not self.manual_picks_dict[self.trace_idx]:
            QMessageBox.information(self, "No Picks", "No manual picks to analyze for this trace.")
            return
        picks = self.manual_picks_dict[self.trace_idx]
        dlg = ReflectionCoeffDialog(self.trace, self.sample_depths, picks, parent=self)
        dlg.exec_()
# --------------------------- MAIN GUI APPLICATION ---------------------------
class ModernHorizonPickerGUI(QMainWindow):
    """
    Main QMainWindow for the SBP Horizon Picker GUI.
    Provides: SEG-Y loading, automatic picking, interactive plots,
    waveform viewer, export, and attribute analysis.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SBP Horizon Picker GUI")
        self.setGeometry(50, 50, 1400, 900)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # State variables
        self.seismic_data = None
        self.sample_depths = None
        self.horizon_picks = []
        self.grouped_horizons = []
        self.local_orientations = []
        self.envelope_smoothed = None
        self.phase = None
        self.worker = None
        self.cpu_count = multiprocessing.cpu_count()
        self.manual_picks_dict = {}
        self.initUI()

    def initUI(self):
        """
        Build the main window: controls panel, main plot panel, and layout.
        """
        main_layout = QHBoxLayout(self.central_widget)
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setAlignment(Qt.AlignTop)
        # Controls for data loading and picking parameters
        self.info_label = QLabel()
        controls_layout.addWidget(self.info_label)
        self.btn_load = QPushButton("Load SEG-Y")
        self.btn_load.clicked.connect(self.load_segy)
        controls_layout.addWidget(self.btn_load)
        controls_layout.addWidget(QLabel("Envelope Threshold Ratio"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.01, 1.0)
        self.thresh_spin.setSingleStep(0.01)
        self.thresh_spin.setValue(0.25)
        self.thresh_spin.setToolTip("Ratio of max envelope for picking peaks")
        controls_layout.addWidget(self.thresh_spin)
        controls_layout.addWidget(QLabel("Smoothing Sigma"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 20.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(8)
        self.sigma_spin.setToolTip("Gaussian sigma for envelope smoothing")
        controls_layout.addWidget(self.sigma_spin)
        controls_layout.addWidget(QLabel("Min Distance"))
        self.mindist_spin = QSpinBox()
        self.mindist_spin.setRange(1, 50)
        self.mindist_spin.setValue(10)
        self.mindist_spin.setToolTip("Minimum distance between peaks")
        controls_layout.addWidget(self.mindist_spin)
        controls_layout.addWidget(QLabel("Max Horizons/Trace"))
        self.maxh_spin = QSpinBox()
        self.maxh_spin.setRange(1, 10)
        self.maxh_spin.setValue(2)
        self.maxh_spin.setToolTip("Maximum horizons per trace")
        controls_layout.addWidget(self.maxh_spin)
        controls_layout.addWidget(QLabel("Ignore Top Samples"))
        self.min_sample_spin = QSpinBox()
        self.min_sample_spin.setRange(0, 100)
        self.min_sample_spin.setValue(10)
        self.min_sample_spin.setToolTip("Ignore top N samples in picking")
        controls_layout.addWidget(self.min_sample_spin)
        controls_layout.addWidget(QLabel("Max Sample Index"))
        self.max_sample_spin = QSpinBox()
        self.max_sample_spin.setRange(1, 10000)
        self.max_sample_spin.setValue(500)
        self.max_sample_spin.setToolTip("Restrict horizon picking up to this sample index (depth/TWT)")
        controls_layout.addWidget(self.max_sample_spin)
        controls_layout.addWidget(QLabel("DBSCAN eps"))
        self.dbscan_eps_spin = QDoubleSpinBox()
        self.dbscan_eps_spin.setRange(1, 100)
        self.dbscan_eps_spin.setSingleStep(1)
        self.dbscan_eps_spin.setValue(40)
        self.dbscan_eps_spin.setToolTip("DBSCAN: max sample distance for clustering")
        controls_layout.addWidget(self.dbscan_eps_spin)
        controls_layout.addWidget(QLabel("DBSCAN min_samples"))
        self.dbscan_min_samples_spin = QSpinBox()
        self.dbscan_min_samples_spin.setRange(1, 20)
        self.dbscan_min_samples_spin.setValue(3)
        self.dbscan_min_samples_spin.setToolTip("DBSCAN: minimum samples for cluster")
        controls_layout.addWidget(self.dbscan_min_samples_spin)
        controls_layout.addWidget(QLabel("Orientation Window"))
        self.orientation_win_spin = QSpinBox()
        self.orientation_win_spin.setRange(3, 21)
        self.orientation_win_spin.setSingleStep(2)
        self.orientation_win_spin.setValue(7)
        self.orientation_win_spin.setToolTip("Window size for dip/azimuth estimation (odd)")
        controls_layout.addWidget(self.orientation_win_spin)
        controls_layout.addWidget(QLabel("n_jobs (cores)"))
        self.njobs_spin = QSpinBox()
        self.njobs_spin.setRange(-1, self.cpu_count)
        self.njobs_spin.setValue(-1)
        self.njobs_spin.setToolTip("Number of parallel jobs for orientation")
        controls_layout.addWidget(self.njobs_spin)
        self.btn_pick = QPushButton("Pick Horizons")
        self.btn_pick.clicked.connect(self.pick_and_plot)
        controls_layout.addWidget(self.btn_pick)
        self.progress = QProgressBar()
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        controls_layout.addWidget(self.progress)
        controls_layout.addSpacing(20)
        self.btn_show_dip_az = QPushButton("Show Dip/Azimuth/Rose")
        self.btn_show_dip_az.clicked.connect(self.show_dip_azimuth_dialog)
        controls_layout.addWidget(self.btn_show_dip_az)
        self.btn_waveform = QPushButton("Show Waveform")
        self.btn_waveform.clicked.connect(self.plot_waveform_dialog)
        controls_layout.addWidget(self.btn_waveform)
        # Export automatic picks to .xlsx
        self.btn_export_auto_xlsx = QPushButton("Export Automatic Picks to .xlsx")
        self.btn_export_auto_xlsx.clicked.connect(self.export_automatic_picks_to_xlsx)
        controls_layout.addWidget(self.btn_export_auto_xlsx)
        controls_layout.addStretch()
        # Main panel: seismic plots and controls
        main_panel = QWidget()
        panel_layout = QVBoxLayout(main_panel)
        self.fig, self.axs = plt.subplots(3, 1, figsize=(12, 16))
        self.canvas = FigureCanvas(self.fig)
        panel_layout.addWidget(self.canvas)
        self.save_main_btn = QPushButton("Save Displayed Graphs (PNG, 900 DPI)")
        self.save_main_btn.clicked.connect(self.save_main_fig)
        panel_layout.addWidget(self.save_main_btn)
        main_layout.addWidget(controls_widget)
        main_layout.addWidget(main_panel)
        main_layout.setStretch(1, 1)

    def load_segy(self):
        """
        Load a SEG-Y file and extract trace data and sample depths.
        Updates main window state and resets picks/attributes.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open SEG-Y File", "",
            "SEG-Y Files (*.sgy *.segy);;All Files (*)", options=options)
        if file_name:
            try:
                with segyio.open(file_name, "r", ignore_geometry=True) as f:
                    num_traces = f.tracecount
                    num_samples = f.samples.size
                    data = np.empty((num_traces, num_samples), dtype=np.float32)
                    for i in range(num_traces):
                        data[i, :] = f.trace[i]
                    depths = np.array(f.samples)
                self.seismic_data = data
                dt_ms = None
                if self.sample_depths is not None and len(self.sample_depths) > 1:
                    dt_ms = float(self.sample_depths[1] - self.sample_depths[0])
                else:
                    dt_ms = 1.0
                self.info_label.setText(
                    f"Loaded SBP data shape: {self.seismic_data.shape}, sample interval: {dt_ms:.3f} ms")
                self.sample_depths = depths
                self.horizon_picks = []
                self.grouped_horizons = []
                self.local_orientations = []
                self.envelope_smoothed = None
                self.phase = None
                self.plot_all()
                self.max_sample_spin.setMaximum(num_samples - 1)
                self.max_sample_spin.setValue(num_samples - 1)
                self.manual_picks_dict = {}
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load SEG-Y file:\n{e}")

    def pick_and_plot(self):
        """
        Launch horizon picking in a background worker, using current parameters.
        """
        if self.seismic_data is None:
            QMessageBox.warning(self, "No Data", "Please load a SEG-Y file first.")
            return
        params = {
            'env_thresh_ratio': self.thresh_spin.value(),
            'sigma': self.sigma_spin.value(),
            'max_horizons': self.maxh_spin.value(),
            'min_dist': self.mindist_spin.value(),
            'min_sample': self.min_sample_spin.value(),
            'max_sample': self.max_sample_spin.value(),
            'n_jobs': self.njobs_spin.value(),
            'dbscan_eps': self.dbscan_eps_spin.value(),
            'dbscan_min_samples': self.dbscan_min_samples_spin.value(),
            'orientation_window': self.orientation_win_spin.value(),
        }
        self.progress.setValue(0)
        self.btn_pick.setEnabled(False)
        self.worker = HorizonWorker(self.seismic_data, self.sample_depths, params)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._on_worker_result)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_result(self, result):
        """
        Callback for horizon worker: update state and replot.
        """
        self.btn_pick.setEnabled(True)
        self.progress.setValue(100)
        self.horizon_picks = result['picks']
        self.grouped_horizons = result['grouped']
        self.local_orientations = result['orientations']
        self.envelope_smoothed = result['envelope']
        self.phase = result['phase']
        self.plot_all()

    def _on_worker_error(self, msg):
        """
        Callback for horizon worker error: show error dialog.
        """
        self.btn_pick.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error during picking:\n{msg}")

    def plot_all(self):
        """
        Draw all main seismic attribute plots (raw, phase, envelope + Haralick transform).
        """
        self.axs[0].clear()
        if self.seismic_data is not None:
            self.axs[0].imshow(self.seismic_data.T, cmap='gray', aspect='auto')
            self.axs[0].set_title('Raw SBP Profile')
            self.axs[0].set_ylabel('Sample Index (or TWT)')
        self.axs[1].clear()
        if self.phase is not None:
            self.axs[1].imshow(np.cos(self.phase).T, cmap='gray', aspect='auto')
            if self.horizon_picks:
                pick_i, pick_j = zip(*self.horizon_picks)
                self.axs[1].plot(pick_i, pick_j, 'rx', markersize=1, label='Raw picks')
            self.axs[1].set_title('Cosine Phase Image with Horizon Picks')
            self.axs[1].set_ylabel('Sample Index (or TWT)')
        else:
            self.axs[1].set_title('Cosine Phase Image')
            self.axs[1].set_ylabel('Sample Index (or TWT)')
        self.axs[2].clear()
        if self.envelope_smoothed is not None:
            gray_env = envelope_to_grayscale(self.envelope_smoothed)
            self.axs[2].imshow(gray_env.T, cmap='gray', aspect='auto')
            if self.horizon_picks:
                pick_i, pick_j = zip(*self.horizon_picks)
                self.axs[2].plot(pick_i, pick_j, 'rx', markersize=1, label='Raw picks')
            self.axs[2].set_title('Envelope Image (Haralick Transform)')
            self.axs[2].set_xlabel('Trace index')
            self.axs[2].set_ylabel('Sample Index (or TWT)')
        else:
            self.axs[2].set_title('Envelope Image (Haralick Transform)')
            self.axs[2].set_xlabel('Trace index')
            self.axs[2].set_ylabel('Sample Index (or TWT)')
        self.fig.tight_layout()
        self.canvas.draw()

    def show_dip_azimuth_dialog(self):
        """
        Show the Dip/Azimuth/Rose diagram dialog for current horizons.
        """
        if not (self.grouped_horizons and self.local_orientations):
            QMessageBox.information(self, "No Data", "No horizon data to show.")
            return
        dlg = DipAzimuthDialog(self.grouped_horizons, self.local_orientations, self)
        dlg.exec_()

    def plot_waveform_dialog(self):
        """
        Show the waveform dialog for a user-selected trace.
        """
        if self.seismic_data is None:
            QMessageBox.warning(self, "No Data", "Please load a SEG-Y file first.")
            return
        max_trace = self.seismic_data.shape[0] - 1
        trace_idx, ok = QInputDialog.getInt(self, "Plot Waveform", f"Enter trace index (0 - {max_trace}):", 0, 0, max_trace)
        if ok:
            if 0 <= trace_idx <= max_trace:
                try:
                    max_sample_index = self.max_sample_spin.value()
                    dlg = WaveformDialog(self.seismic_data[trace_idx], self.sample_depths, trace_idx, self, self.manual_picks_dict, max_sample_index=max_sample_index)
                    dlg.exec_()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to plot waveform:\n{e}")
            else:
                QMessageBox.warning(self, "Invalid Index", f"Trace index must be between 0 and {max_trace}.")

    def plot_manual_picks_on_section(self):
        """
        Overlay manual picks on the main seismic plot for visual inspection.
        """
        if self.seismic_data is None or not self.manual_picks_dict:
            QMessageBox.information(self, "No Picks",
                                    "No manual picks to display. Use the waveform dialog to pick reflectors first.")
            return
        self.axs[0].clear()
        extent = [0, self.seismic_data.shape[0], 0, self.seismic_data.shape[1]]
        self.axs[0].imshow(self.seismic_data.T, cmap='gray', aspect='auto', extent=extent, origin='upper')
        first = True
        for trc_idx, pick_indices in self.manual_picks_dict.items():
            for idx in pick_indices:
                self.axs[0].plot(trc_idx, idx, 'rx', markersize=1,
                                 label="Manual Pick" if first else "")
                first = False
        self.axs[0].set_title('Raw SBP Profile + Manual Picks')
        self.axs[0].set_ylabel('Sample Index (TWT)')
        self.axs[0].set_xlabel('Trace index')
        self.axs[0].invert_yaxis()
        self.axs[0].legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def save_main_fig(self):
        """
        Export the main (multi-panel) figure as a PNG image.
        """
        fname, _ = QFileDialog.getSaveFileName(self, "Save Main Displayed Graphs", "", "PNG Image (*.png)")
        if fname:
            self.fig.savefig(fname, dpi=900, bbox_inches='tight')

    def export_automatic_picks_to_xlsx(self):
        """
        Export automatic horizon picks to Excel for further analysis.
        """
        if not self.horizon_picks:
            QMessageBox.information(self, "No Picks", "No automatic picks to export.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Export Automatic Picks", "", "Excel Files (*.xlsx)")
        if not fname:
            return
        rows = []
        for trace_idx, sample_idx in self.horizon_picks:
            depth = twt_to_depth(self.sample_depths[sample_idx])
            rows.append({'Trace Index': trace_idx, 'Sample Index': sample_idx, 'Depth (m)': depth})
        df = pd.DataFrame(rows)
        df.to_excel(fname, index=False)
        QMessageBox.information(self, "Export Complete", f"Automatic picks exported to {fname}")

# ---------------------------- MAIN ENTRY POINT ------------------------------
def main():
    """
    Main entry point for launching the application.
    """
    app = QApplication(sys.argv)
    win = ModernHorizonPickerGUI()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
