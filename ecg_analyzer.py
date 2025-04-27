import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal
from skimage import measure
import base64
from io import BytesIO
import logging

def analyze_ecg(image_path):
    """
    Analyze ECG image to extract waveform and perform cardiac analysis
    
    Args:
        image_path (str): Path to the ECG image file
        
    Returns:
        dict: Results containing extracted data and analysis
    """
    try:
        # Load and preprocess the image
        ecg_data = preprocess_image(image_path)
        
        # Extract the waveform
        waveform = extract_waveform(ecg_data)
        
        # Analyze the extracted waveform
        analysis_results = analyze_waveform(waveform)
        
        # Generate visualizations
        visualization = generate_visualization(waveform, analysis_results)
        
        # Combine all results
        results = {
            'waveform': waveform.tolist(),  # Convert to list for JSON serialization
            'heart_rate': analysis_results['heart_rate'],
            'rhythm_analysis': analysis_results['rhythm_analysis'],
            'visualization': visualization,
            'p_wave_present': analysis_results['p_wave_present'],
            'qrs_duration': analysis_results['qrs_duration'],
            'qt_interval': analysis_results['qt_interval'],
            'error': None
        }
        
        return results
    
    except Exception as e:
        logging.error(f"Error in ECG analysis: {str(e)}")
        return {
            'waveform': None,
            'heart_rate': None,
            'rhythm_analysis': None,
            'visualization': None,
            'p_wave_present': None,
            'qrs_duration': None,
            'qt_interval': None,
            'error': str(e)
        }

def preprocess_image(image_path):
    """
    Preprocess the ECG image for analysis
    
    Args:
        image_path (str): Path to the ECG image
        
    Returns:
        numpy.ndarray: Preprocessed image ready for waveform extraction
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load the image. Check if the file exists and is a valid image.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to create binary image
    # Use adaptive thresholding to handle varying lighting conditions
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Remove grid lines and small noise using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Further processing to enhance ECG trace
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    
    return dilated

def extract_waveform(preprocessed_image):
    """
    Extract the ECG waveform from the preprocessed image
    
    Args:
        preprocessed_image (numpy.ndarray): Preprocessed binary image
        
    Returns:
        numpy.ndarray: Extracted waveform signal
    """
    # Find contours of the ECG trace
    contours, _ = cv2.findContours(
        preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # If no contours found, try a different approach
    if not contours:
        # Try row-wise summation to detect the waveform
        projection = np.sum(preprocessed_image, axis=1)
        return projection
    
    # Find the largest contour by area (likely the ECG trace)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask for the largest contour
    mask = np.zeros_like(preprocessed_image)
    cv2.drawContours(mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    
    # Extract points along the contour to form the waveform
    # We'll create a waveform by finding the highest white pixel in each column
    height, width = preprocessed_image.shape
    waveform = np.zeros(width)
    
    for x in range(width):
        col = mask[:, x]
        # Find the y-coordinates of white pixels in this column
        white_pixels = np.where(col > 0)[0]
        if len(white_pixels) > 0:
            # Take the highest point (minimum y-value)
            waveform[x] = np.min(white_pixels)
        else:
            # If no white pixels, use the previous value or a default
            waveform[x] = waveform[x-1] if x > 0 else height
    
    # Invert the waveform (since y increases downward in images)
    waveform = height - waveform
    
    # Normalize the waveform to a range of 0-1
    if np.max(waveform) > np.min(waveform):
        waveform = (waveform - np.min(waveform)) / (np.max(waveform) - np.min(waveform))
    
    # Apply smoothing to reduce noise
    waveform = signal.savgol_filter(waveform, 15, 3)
    
    return waveform

def analyze_waveform(waveform):
    """
    Analyze the ECG waveform to extract cardiac parameters
    
    Args:
        waveform (numpy.ndarray): Extracted ECG waveform
        
    Returns:
        dict: Analysis results including heart rate and rhythm analysis
    """
    # Find R peaks (the highest points in the QRS complex)
    # Use peak finding algorithm from scipy
    peaks, _ = signal.find_peaks(waveform, height=0.5, distance=30)
    
    results = {}
    
    # Calculate heart rate if we have at least 2 R peaks
    if len(peaks) >= 2:
        # Assume the x-axis units are samples at 250 Hz (typical for ECG)
        # This is an approximation since we don't know the actual time scale
        sample_rate = 250
        # Calculate average RR interval in seconds
        rr_intervals = np.diff(peaks) / sample_rate
        # Heart rate = 60 / average RR interval in seconds
        avg_rr = np.mean(rr_intervals)
        heart_rate = 60 / avg_rr if avg_rr > 0 else 0
        results['heart_rate'] = round(heart_rate)
        
        # Simple rhythm analysis based on RR interval variability
        rr_variability = np.std(rr_intervals) / avg_rr if avg_rr > 0 else 0
        
        if rr_variability < 0.05:
            rhythm = "Regular rhythm"
        elif rr_variability < 0.15:
            rhythm = "Mild irregularity, possible sinus arrhythmia"
        else:
            rhythm = "Irregular rhythm, possible atrial fibrillation"
            
        results['rhythm_analysis'] = rhythm
    else:
        results['heart_rate'] = None
        results['rhythm_analysis'] = "Insufficient peaks detected for rhythm analysis"
    
    # Detect P waves (typically occur before R peaks)
    # This is a simplified approach
    p_waves_detected = 0
    if len(peaks) > 0:
        for peak in peaks:
            if peak > 20:  # Ensure we have enough window to look back
                # Look for a small bump before the R peak
                window = waveform[peak-20:peak-5]
                p_peak, _ = signal.find_peaks(window, height=0.1, distance=5)
                if len(p_peak) > 0:
                    p_waves_detected += 1
    
    # Calculate P wave presence ratio
    if len(peaks) > 0:
        p_wave_ratio = p_waves_detected / len(peaks)
        results['p_wave_present'] = "Yes" if p_wave_ratio > 0.5 else "No"
    else:
        results['p_wave_present'] = "Unknown"
    
    # Estimate QRS duration (simplified)
    # We'll measure the width of the peaks at half-maximum height
    if len(peaks) > 0:
        qrs_widths = []
        for peak in peaks:
            if peak > 10 and peak < len(waveform) - 10:
                # Define the region around the peak
                region = waveform[peak-10:peak+10]
                # Find the half height
                half_height = (np.max(region) + np.min(region)) / 2
                # Find points crossing the half height
                crossings = np.where(np.diff(region > half_height))[0]
                if len(crossings) >= 2:
                    # Calculate width in samples
                    width = crossings[-1] - crossings[0]
                    # Convert to milliseconds (assuming 250 Hz)
                    width_ms = width * (1000 / 250)
                    qrs_widths.append(width_ms)
        
        if qrs_widths:
            results['qrs_duration'] = f"{np.mean(qrs_widths):.1f} ms"
        else:
            results['qrs_duration'] = "Could not determine"
    else:
        results['qrs_duration'] = "Could not determine"
    
    # Estimate QT interval (simplified)
    if len(peaks) > 0:
        qt_intervals = []
        for peak in peaks:
            if peak < len(waveform) - 50:
                # Look for T wave end after R peak
                t_region = waveform[peak:peak+50]
                # T wave usually ends at a local minimum
                t_end_candidates, _ = signal.find_peaks(-t_region)
                if len(t_end_candidates) > 0:
                    # Take the most prominent minimum
                    t_end = t_end_candidates[-1]
                    # QT interval in samples
                    qt = t_end
                    # Convert to milliseconds (assuming 250 Hz)
                    qt_ms = qt * (1000 / 250)
                    qt_intervals.append(qt_ms)
        
        if qt_intervals:
            results['qt_interval'] = f"{np.mean(qt_intervals):.1f} ms"
        else:
            results['qt_interval'] = "Could not determine"
    else:
        results['qt_interval'] = "Could not determine"
    
    return results

def generate_visualization(waveform, analysis_results):
    """
    Generate visualizations of the ECG waveform and analysis
    
    Args:
        waveform (numpy.ndarray): The extracted ECG waveform
        analysis_results (dict): Results of the waveform analysis
    
    Returns:
        str: Base64 encoded image of the visualization
    """
    # Set the figure style for dark theme compatibility
    plt.style.use('dark_background')
    
    # Create figure with multiple subplots
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the waveform
    x = np.arange(len(waveform))
    ax.plot(x, waveform, 'c-', linewidth=1.5)
    
    # Mark R peaks if available
    peaks, _ = signal.find_peaks(waveform, height=0.5, distance=30)
    if len(peaks) > 0:
        ax.plot(peaks, waveform[peaks], 'ro', markersize=6)
    
    # Add annotations
    heart_rate = analysis_results.get('heart_rate')
    if heart_rate:
        ax.annotate(f"Heart Rate: {heart_rate} BPM", 
                   xy=(0.02, 0.95), xycoords='axes fraction',
                   color='white', fontsize=12)
    
    rhythm = analysis_results.get('rhythm_analysis')
    if rhythm:
        ax.annotate(f"Rhythm: {rhythm}", 
                   xy=(0.02, 0.90), xycoords='axes fraction',
                   color='white', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('ECG Waveform Analysis', fontsize=14)
    
    # Add legend
    ax.legend(['ECG Signal', 'R Peaks'], loc='lower right')
    
    # Tight layout
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return image_base64
