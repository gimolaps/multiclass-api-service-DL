const fileInput = document.getElementById("fileInput");
const imagePreview = document.getElementById("imagePreview");
const previewBox = document.getElementById("previewBox");
const resultBox = document.getElementById("result");
const predictButton = document.getElementById("predictButton");


fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];

    resultBox.classList.add("hidden");
    resultBox.innerHTML = "";

    if (!file) {
        previewBox.classList.add("hidden");
        imagePreview.src = "";
        return;
    }

    const imageUrl = URL.createObjectURL(file);
    imagePreview.src = imageUrl;
    previewBox.classList.remove("hidden");
});


async function predictImage() {
    const file = fileInput.files[0];

    if (!file) {
        showError("No image selected", "Choose JPG, PNG or WEBP image first.");
        return;
    }

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!allowedTypes.includes(file.type)) {
        showError("Wrong file format", "Allowed formats: JPG, JPEG, PNG, WEBP.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    predictButton.disabled = true;
    predictButton.textContent = "Running model...";

    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `
        <div class="result-title">Processing image...</div>
        <div class="result-subtitle">The CNN model is calculating top-5 predictions.</div>
    `;

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            showError("API request failed", data.detail || JSON.stringify(data));
            return;
        }

        renderPredictions(data);

    } catch (error) {
        showError("Request failed", String(error));
    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Image";
    }
}


function renderPredictions(data) {
    const topPrediction = data.top_prediction;
    const predictions = data.predictions;

    let predictionsHtml = "";

    predictions.forEach((item, index) => {
        predictionsHtml += `
            <div class="prediction-item">
                <div>
                    <div class="prediction-class">${index + 1}. ${escapeHtml(item.class_name)}</div>
                </div>
                <div class="prediction-confidence">${item.confidence}%</div>
            </div>
        `;
    });

    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `
        <div class="result-title">Top prediction: ${escapeHtml(topPrediction.class_name)}</div>
        <div class="result-subtitle">Confidence: ${topPrediction.confidence}%</div>
        <div class="prediction-list">
            ${predictionsHtml}
        </div>
    `;
}


function showError(title, message) {
    resultBox.classList.remove("hidden");
    resultBox.innerHTML = `
        <div class="result-title">${escapeHtml(title)}</div>
        <div class="result-subtitle">${escapeHtml(message)}</div>
    `;
}


function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}