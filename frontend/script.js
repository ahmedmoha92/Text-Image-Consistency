// --- Éléments du DOM ---
const dropzone = document.getElementById('dropzone');
const dropzoneText = document.getElementById('dropzone-text');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const btnSpinner = document.getElementById('btnSpinner');
const resultDiv = document.getElementById('result');
const textInput = document.getElementById('textInput');

// --- Upload : cliquer sur la zone ---
dropzone.addEventListener('click', () => imageInput.click());

// --- Upload : glisser-déposer ---
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = '#007bff';
});
dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = imageInput.files.length ? '#28a745' : '#ccc';
});
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) {
        imageInput.files = e.dataTransfer.files;
        showPreview(e.dataTransfer.files[0]);
    }
});

// --- Upload : sélection de fichier ---
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        showPreview(e.target.files[0]);
    }
});

// --- Aperçu de l'image ---
function showPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.add('visible');
        dropzone.classList.add('has-image');
        dropzoneText.textContent = file.name;
    };
    reader.readAsDataURL(file);
}

// --- Gestion de l'état du bouton ---
function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    btnText.textContent = isLoading ? 'Analyse en cours...' : 'Prédire';
    btnSpinner.hidden = !isLoading;
}

// --- Prédiction ---
predictBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    const file = imageInput.files[0];

    if (!text || !file) {
        resultDiv.innerHTML = '<div class="result-card error">Veuillez fournir un texte et une image.</div>';
        return;
    }

    setLoading(true);
    resultDiv.innerHTML = '';

    const reader = new FileReader();
    reader.onloadend = async function () {
        const base64Image = reader.result.split(',')[1];
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text, image: base64Image }),
            });
            if (!response.ok) {
                const err = await response.json().catch(() => ({}));
                throw new Error(err.error || `Erreur serveur (${response.status})`);
            }
            const data = await response.json();
            const cssClass = data.prediction === 'coherent' ? 'coherent' : 'incoherent';
            resultDiv.innerHTML = `<div class="result-card ${cssClass}">
                <strong>Prédiction : ${data.prediction}</strong><br>
                Confiance : ${(data.confidence * 100).toFixed(2)}%
            </div>`;
        } catch (err) {
            resultDiv.innerHTML = `<div class="result-card error">Erreur : ${err.message}</div>`;
        } finally {
            setLoading(false);
        }
    };
    reader.readAsDataURL(file);
});
