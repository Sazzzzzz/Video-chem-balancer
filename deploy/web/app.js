import init, { balance_equation } from "./pkg/chem_balancer.js";

const equationInput = document.getElementById("equation-input");
const balanceBtn = document.getElementById("balance-btn");
const exampleBtn = document.getElementById("example-btn");
const resultContent = document.getElementById("result-content");
const errorArea = document.getElementById("error-area");
const copyBtn = document.getElementById("copy-btn");
const solutionsArea = document.getElementById("solutions-area");
const solutionsList = document.getElementById("solutions-list");
const themeToggle = document.getElementById("theme-toggle");
const exampleChips = document.querySelectorAll(".example-chip");

// Theme
const savedTheme = localStorage.getItem("theme") || "light";
document.documentElement.setAttribute("data-theme", savedTheme);
themeToggle.addEventListener("click", () => {
    const next =
        document.documentElement.getAttribute("data-theme") === "dark"
            ? "light"
            : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
});

// WASM init
let wasmReady = false;
async function initWasm() {
    try {
        await init();
        wasmReady = true;
    } catch (err) {
        resultContent.innerHTML =
            '<p class="placeholder-text" style="color:var(--error)">Failed to load engine. Refresh the page.</p>';
    }
}

function balance(raw) {
    if (!wasmReady) {
        showError("Engine loading, try again.");
        return;
    }
    if (!raw.trim()) {
        showError("Enter a chemical equation.");
        return;
    }

    const eq = raw
        .replace(/→/g, "==")
        .replace(/⇌/g, "==")
        .replace(/⇄/g, "==")
        .replace(/->/g, "==")
        .replace(/(?<!-)=(?!=)/g, "==");

    clear();

    try {
        const data = JSON.parse(balance_equation(eq));
        if (data.error) {
            showError(data.error);
            return;
        }
        if (!data.balanced?.length) {
            resultContent.innerHTML =
                '<p class="placeholder-text">No solution found.</p>';
            return;
        }

        const main = data.balanced[0].replace(/\s*==\s*/g, " → ");
        resultContent.innerHTML = `<div class="balanced-equation">${main}</div>`;
        copyBtn.style.display = "flex";
        copyBtn.dataset.copyText = main;

        if (data.balanced.length > 1) {
            solutionsArea.style.display = "block";
            solutionsList.innerHTML = data.balanced
                .map((s, i) => {
                    const f = s.replace(/\s*==\s*/g, " → ");
                    return `<div class="solution-item"><span class="solution-index">#${i + 1}</span><span>${f}</span></div>`;
                })
                .join("");
        }
    } catch (err) {
        showError("Error: " + err.message);
    }
}

function showError(msg) {
    errorArea.style.display = "block";
    errorArea.textContent = msg;
    resultContent.innerHTML =
        '<p class="placeholder-text">Your balanced equation will appear here</p>';
    copyBtn.style.display = "none";
    solutionsArea.style.display = "none";
}

function clear() {
    errorArea.style.display = "none";
    errorArea.textContent = "";
    copyBtn.style.display = "none";
    solutionsArea.style.display = "none";
    solutionsList.innerHTML = "";
}

// Copy
copyBtn.addEventListener("click", () => {
    const text = copyBtn.dataset.copyText;
    if (!text) return;
    navigator.clipboard
        .writeText(text)
        .then(() => {
            const toast = Object.assign(document.createElement("div"), {
                className: "toast",
                textContent: "Copied!",
            });
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 1800);
        })
        .catch(() => {});
});

// Events
balanceBtn.addEventListener("click", () => balance(equationInput.value));
equationInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        balance(equationInput.value);
    }
});

exampleBtn.addEventListener("click", () => {
    const pool = [
        "H2 + O2 = H2O",
        "Fe + O2 = Fe2O3",
        "CH4 + O2 = CO2 + H2O",
        "KMnO4 + HCl = KCl + MnCl2 + H2O + Cl2",
        "C6H12O6 + O2 = CO2 + H2O",
        "NH4ClO4 + HNO3 + HCl + H2O -> H5ClO6 + N2O + NO + NO2 + Cl2",
    ];
    const pick = pool[Math.floor(Math.random() * pool.length)];
    equationInput.value = pick;
    balance(pick);
});

exampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
        equationInput.value = chip.dataset.equation;
        balance(chip.dataset.equation);
    });
});

initWasm();
