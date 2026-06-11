document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = form.querySelector('.submit-btn');
            const originalBtnContent = submitBtn.innerHTML;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running Analysis...';
            submitBtn.disabled = true;
            
            // Collect form data
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
            // Client-side validation to ensure no empty fields
            for (const [key, value] of Object.entries(data)) {
                if (value === '' || value === null) {
                    resultContainer.style.display = 'block';
                    resultContainer.className = 'result-container result-error';
                    resultContainer.innerHTML = `
                        <div class="result-title">Validation Error</div>
                        <div>Please fill out all required fields. Missing: ${key.replace('_', ' ')}</div>
                    `;
                    submitBtn.innerHTML = originalBtnContent;
                    submitBtn.disabled = false;
                    return;
                }
            }
            
            // Determine endpoint based on form ID or page
            let endpoint = '';
            if (window.location.pathname.includes('crop')) {
                endpoint = '/api/predict-crop';
            } else if (window.location.pathname.includes('fertilizer')) {
                endpoint = '/api/predict-fertilizer';
            } else if (window.location.pathname.includes('type')) {
                endpoint = '/api/predict-type';
            }
            
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });
                
                const result = await response.json();
                
                resultContainer.style.display = 'block';
                
                if (result.success) {
                    resultContainer.className = 'result-container result-success';
                    resultContainer.innerHTML = `
                        <div class="result-title">Analysis Success</div>
                        <div class="result-value-wrapper">
                            <div class="result-label">Recommended Match</div>
                            <div class="result-value">${result.prediction}</div>
                        </div>
                        <div class="confidence-wrapper">
                            <div class="confidence-info">
                                <span>Confidence Rating</span>
                                <span>${result.confidence}%</span>
                            </div>
                            <div class="confidence-bar-bg">
                                <div class="confidence-bar-fill" id="conf-bar-fill"></div>
                            </div>
                        </div>
                    `;
                    
                    // Trigger the transition width animation in next event loop tick
                    setTimeout(() => {
                        const fillElement = document.getElementById('conf-bar-fill');
                        if (fillElement) {
                            fillElement.style.width = `${result.confidence}%`;
                        }
                    }, 50);
                } else {
                    resultContainer.className = 'result-container result-error';
                    resultContainer.innerHTML = `
                        <div class="result-title">Server Error</div>
                        <div>${result.error}</div>
                    `;
                }
                
            } catch (error) {
                resultContainer.style.display = 'block';
                resultContainer.className = 'result-container result-error';
                resultContainer.innerHTML = `
                    <div class="result-title">Connection Failed</div>
                    <div>${error.message}</div>
                `;
            } finally {
                submitBtn.innerHTML = originalBtnContent;
                submitBtn.disabled = false;
            }
        });
    }
});
