import init, { balance } from "./pkg/chem_balancer.js";

const equationInput = document.getElementById("equation-input");
const balanceBtn = document.getElementById("balance-btn");
const loadingArea = document.getElementById("loading-area");
const errorArea = document.getElementById("error-area");
const resultArea = document.getElementById("result-area");
const matrixBody = document.getElementById("matrix-body");
const hilbertBody = document.getElementById("hilbert-body");
const panelCopyBtns = document.querySelectorAll(".panel-copy-btn");
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
        matrixBody.innerHTML =
            '<p class="placeholder-text" style="color:var(--error)">无法加载引擎。请刷新页面。</p>';
        hilbertBody.innerHTML =
            '<p class="placeholder-text" style="color:var(--error)">无法加载引擎。请刷新页面。</p>';
    }
}

// ---- Equation highlighting ----

function highlightEquation(tokens) {
    if (typeof tokens === "string") return escapeHtml(tokens);

    return tokens
        .map((token) => {
            let val = escapeHtml(token.value);
            let cssClass = "";

            switch (token.ttype) {
                case "Element":
                    cssClass = "hl-element";
                    break;
                case "Number":
                    cssClass = "hl-number";
                    break;
                case "Equals":
                    cssClass = "hl-separator";
                    val = "→";
                    break;
                case "Plus":
                case "Minus":
                    cssClass = "hl-separator";
                    break;
            }

            if (cssClass) {
                return `<span class="${cssClass}">${val}</span>`;
            }
            return val;
        })
        .join("");
}

// ---- Core balance logic ----

function balanceEquation(raw) {
    if (!wasmReady) {
        showError("引擎仍在加载，请稍候...");
        return;
    }
    if (!raw.trim()) {
        showError("请输入化学方程式");
        return;
    }

    // Normalize separators to ==
    const eq = raw
        .replace(/→/g, "==")
        .replace(/⇌/g, "==")
        .replace(/⇄/g, "==")
        .replace(/->/g, "==")
        .replace(/(?<!-)=(?!=)/g, "==");

    hideResults();
    showLoading();

    // Yield to the event loop so the spinner paints before blocking WASM
    requestAnimationFrame(() => {
        setTimeout(() => {
            try {
                const matrixData = JSON.parse(balance(eq, "matrix"));
                const dualSolutionsNode =
                    document.querySelector(".dual-solutions");
                const mPanelTitle = document.querySelector(
                    "#matrix-panel .solution-panel-title",
                );
                const hPanel = document.getElementById("hilbert-panel");

                hideLoading();

                if (
                    matrixData.error &&
                    matrixData.error.includes("Failed to parse")
                ) {
                    showError(matrixData.error);
                    return;
                }

                if (matrixData.balanced && matrixData.balanced.length === 1) {
                    // Only 1 solution => short circuit
                    renderPanel("matrix", matrixData);
                    dualSolutionsNode.classList.add("single-col");
                    hPanel.style.display = "none";
                    if (mPanelTitle) {
                        mPanelTitle.innerHTML = "矩阵法 / Hilbert基 解";
                    }
                    resultArea.style.display = "grid";
                } else {
                    const hilbertData = JSON.parse(balance(eq, "hilbert"));
                    renderPanel("matrix", matrixData);
                    renderPanel("hilbert", hilbertData);

                    dualSolutionsNode.classList.remove("single-col");
                    hPanel.style.display = ""; // reset to CSS default
                    if (mPanelTitle) {
                        mPanelTitle.innerHTML = "矩阵法 解";
                    }
                    resultArea.style.display = "grid";
                }
            } catch (err) {
                hideLoading();
                showError("意外错误: " + err.message);
            }
        }, 50);
    });
}

function renderPanel(method, data) {
    const body = method === "matrix" ? matrixBody : hilbertBody;

    if (data.error) {
        body.innerHTML = `<div class="panel-error">${escapeHtml(data.error)}</div>`;
        return;
    }

    if (!data.balanced || data.balanced.length === 0) {
        body.innerHTML = '<p class="placeholder-text">未找到求解</p>';
        return;
    }

    const solutions = data.balanced;

    if (solutions.length === 1) {
        body.innerHTML = `<div class="balanced-equation">${highlightEquation(solutions[0])}</div>`;
    } else {
        body.innerHTML = solutions
            .map((s, i) => {
                return `<div class="solution-item">
                    <span class="solution-index">#${i + 1}</span>
                    <span>${highlightEquation(s)}</span>
                </div>`;
            })
            .join("");
    }
}

// ---- UI helpers ----

function showLoading() {
    loadingArea.style.display = "flex";
    errorArea.style.display = "none";
    resultArea.style.display = "none";
}

function hideLoading() {
    loadingArea.style.display = "none";
}

function hideResults() {
    errorArea.style.display = "none";
    resultArea.style.display = "none";
}

function showError(msg) {
    hideLoading();
    errorArea.style.display = "block";
    errorArea.textContent = msg;
    resultArea.style.display = "none";
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

// ---- Copy buttons ----

panelCopyBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
        const panel = btn.dataset.panel;
        const body = panel === "matrix" ? matrixBody : hilbertBody;

        let text;
        const items = body.querySelectorAll(".solution-item span:last-child");
        const single = body.querySelector(".balanced-equation");

        if (items.length > 0) {
            text = Array.from(items)
                .map((span, i) => `${i + 1}. ${span.textContent}`)
                .join("\n");
        } else if (single) {
            text = single.textContent;
        } else {
            return;
        }

        navigator.clipboard
            .writeText(text)
            .then(() => {
                showToast("已复制");
            })
            .catch(() => {});
    });
});

function showToast(msg) {
    const toast = Object.assign(document.createElement("div"), {
        className: "toast",
        textContent: msg,
    });
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 1800);
}

// ---- Event listeners ----

balanceBtn.addEventListener("click", () =>
    balanceEquation(equationInput.value),
);
equationInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        balanceEquation(equationInput.value);
    }
});

exampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
        equationInput.value = chip.dataset.equation;
        balanceEquation(chip.dataset.equation);
    });
});

// ---- Start ----

initWasm();
