async function classify() {
    const input = document.getElementById("requirement");
    const resultEl = document.getElementById("result");
    const text = input.value.trim();

    if (!text) {
        resultEl.textContent = "Please enter a requirement.";
        resultEl.style.color = "#ff6b6b";
        return;
    }

    try {
        const response = await fetch("/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ requirement: text }),
        });

        if (!response.ok) {
            const error = await response.json();
            resultEl.textContent = error.error || "Error classifying requirement.";
            resultEl.style.color = "#ff6b6b";
            return;
        }

        const data = await response.json();
        resultEl.textContent = `Requirement classified as: ${data.category.replace('_', ' ')}`;
        resultEl.style.color = "#ffd8b1";

        input.value = "";

    } catch (error) {
        resultEl.textContent = "Failed to classify requirement.";
        resultEl.style.color = "#ff6b6b";
    }
}
