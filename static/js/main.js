// ECG Analyzer - Main JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // Handle form submission with loading indicator
    const ecgForm = document.getElementById('ecgForm');
    const submitBtn = document.getElementById('submitBtn');
    
    if (ecgForm) {
        ecgForm.addEventListener('submit', function(e) {
            // Get the file input
            const fileInput = document.getElementById('ecg_image');
            
            // Check if a file is selected
            if (fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select an ECG image file.');
                return;
            }
            
            // Check file type
            const file = fileInput.files[0];
            if (!file.type.match('image/png')) {
                e.preventDefault();
                alert('Only PNG files are allowed.');
                return;
            }
            
            // Show loading state
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
            submitBtn.disabled = true;
        });
    }
    
    // File input change event - preview image
    const fileInput = document.getElementById('ecg_image');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                // Reset submit button if it was in loading state
                if (submitBtn.disabled) {
                    submitBtn.innerHTML = 'Analyze ECG';
                    submitBtn.disabled = false;
                }
            }
        });
    }
    
    // Enable tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
