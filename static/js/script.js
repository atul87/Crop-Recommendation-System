document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const submitBtn = form.querySelector('.submit-btn');
            const originalBtnText = submitBtn.textContent;
            submitBtn.textContent = 'Predicting...';
            submitBtn.disabled = true;
            
            // Collect form data
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            
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
                        <div class="result-title">Prediction Successful!</div>
                        <div>Recommended: <span class="result-value">${result.prediction}</span></div>
                        <div>Confidence: ${result.confidence}%</div>
                    `;
                } else {
                    resultContainer.className = 'result-container result-error';
                    resultContainer.innerHTML = `
                        <div class="result-title">Error</div>
                        <div>${result.error}</div>
                    `;
                }
                
            } catch (error) {
                resultContainer.style.display = 'block';
                resultContainer.className = 'result-container result-error';
                resultContainer.innerHTML = `
                    <div class="result-title">Network Error</div>
                    <div>${error.message}</div>
                `;
            } finally {
                submitBtn.textContent = originalBtnText;
                submitBtn.disabled = false;
            }
        });
    }
});
