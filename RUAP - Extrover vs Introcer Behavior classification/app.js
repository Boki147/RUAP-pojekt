/**
 * app.js — PersonaLab Frontend Logic
 *
 * - Form input: slideri i toggle gumbi
 * - POST /api/predict → Azure ML → sprema u SQLite → prikazuje rezultat
 * - GET /api/charts   → Pandas + Plotly Express JSON → renderira 3 grafikona
 */

const API_BASE = "http://localhost:5000/api";

const $ = id => document.getElementById(id);

// ─── Range slideri — live prikaz vrijednosti ──────────────────────────────────
document.querySelectorAll('input[type="range"]').forEach(slider => {
  const display = document.getElementById(slider.id + "_val");
  if (display) {
    display.textContent = slider.value;
    slider.addEventListener("input", () => {
      display.textContent = slider.value;
    });
  }
});

// ─── Toggle gumbi (Da/Ne) ──────────────────────────────────────────────────────
document.querySelectorAll(".toggle-group").forEach(group => {
  group.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
});

function getToggleValue(groupId) {
  const activeBtn = document.querySelector(`#${groupId} .toggle-btn.active`);
  return activeBtn ? parseInt(activeBtn.dataset.value) : 0;
}

// ─── Submit forme ──────────────────────────────────────────────────────────────
const form = $("predict-form");
if (form) {
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await submitPrediction();
  });
}

async function submitPrediction() {
  const submitBtn  = $("submit-btn");
  const resultCard = $("result-card");

  const payload = {
    time_spent_alone:     parseFloat($("time_spent_alone").value),
    stage_fear:           getToggleValue("stage_fear_group"),
    social_event_attend:  parseFloat($("social_event_attend").value),
    going_outside:        parseFloat($("going_outside").value),
    drained_after_social: getToggleValue("drained_group"),
    friends_circle_size:  parseFloat($("friends_circle_size").value),
    post_frequency:       parseFloat($("post_frequency").value)
  };

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="spinner"></span>Analizujem...';
  resultCard.classList.remove('visible');

  try {
    const res  = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Greška pri komunikaciji s API-jem");

    showResult(data.prediction, data.confidence);

    // Nakon uspješne predikcije — osvježi Plotly grafikone s novim podatkom
    await loadPlotlyCharts();

  } catch (err) {
    resultCard.className    = "result-card error-card";
    resultCard.classList.add('visible');
    resultCard.innerHTML = `
      <div class="result-emoji">⚠️</div>
      <div class="result-type" style="font-size:1.2rem;">Greška</div>
      <div class="result-desc">${err.message}</div>
      <div class="result-desc" style="margin-top:0.5rem;font-size:0.78rem;opacity:0.7;">
        Provjeri je li Flask server pokrenut na portu 5000.<br>
        <code>python server.py</code>
      </div>
    `;
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = "Otkrij svoju osobnost →";
  }
}

function showResult(prediction, confidence) {
  const card       = $("result-card");
  const isIntrovert = prediction === "Introvert";

  card.className    = `result-card ${isIntrovert ? "introvert" : "extrovert"}`;
  card.classList.add('visible');

  const emoji = isIntrovert ? "🌿" : "🌟";
  const desc  = isIntrovert
    ? "Obnavljaš energiju u miru i samoći. Duboko razmišljanje ti je prirodno."
    : "Energičan/a si u društvu i veseliš se novim vezama i avanturama.";

  const confText = confidence != null
    ? `<div class="result-desc" style="font-size:0.78rem;opacity:0.6;margin-top:0.4rem;">
         Pouzdanost modela: ${(confidence * 100).toFixed(1)}%
       </div>`
    : "";

  card.innerHTML = `
    <div class="result-emoji">${emoji}</div>
    <div class="result-type">${prediction}</div>
    <div class="result-desc">${desc}</div>
    ${confText}
  `;

  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ─── PLOTLY GRAFIKONI ──────────────────────────────────────────────────────────

// Plotly config — bez loga, responsive
const PLOTLY_CFG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["toImage", "sendDataToCloud"]
};

async function loadPlotlyCharts() {
  try {
    const res = await fetch(`${API_BASE}/charts`);
    if (!res.ok) throw new Error("API nedostupan");
    const data = await res.json();

    // Ažuriraj summary stats
    updateSummaryStats(data);

    // Renderiraj sve tri Plotly grafikona iz Flask/Pandas/Plotly JSON-a
    renderPlotly("plotly-dist", data.chart1);
    renderPlotly("plotly-avg",  data.chart2);
    renderPlotly("plotly-time", data.chart3);

  } catch (err) {
    console.warn("Grafikoni — greška:", err.message);
    ["plotly-dist", "plotly-avg", "plotly-time"].forEach(id => {
      const el = $(id);
      if (el) {
        el.innerHTML = `
          <div class="chart-error">
            <span>📡</span>
            <p>Pokreni server za grafikone:</p>
            <code>python server.py</code>
          </div>
        `;
      }
    });
  }
}

/**
 * renderPlotly — prima Plotly figure JSON (data + layout) s Flaska
 * i renderira ga direktno u HTML element pomoću Plotly.react().
 * Plotly.react() je efikasniji od newPlot() — samo ažurira ako postoji.
 */
function renderPlotly(elementId, figureJson) {
  const el = $(elementId);
  if (!el || !figureJson) return;

  // Prilagodi layout za container
  const layout = Object.assign({}, figureJson.layout, {
    autosize: true,
    margin: { t: 45, b: 30, l: 40, r: 20 }
  });

  Plotly.react(el, figureJson.data, layout, PLOTLY_CFG);
}

function updateSummaryStats(data) {
  const totalEl = $("stat-total");
  const introEl = $("stat-intro");
  const extEl   = $("stat-ext");

  if (totalEl) totalEl.textContent = data.total           ?? "—";
  if (introEl) introEl.textContent = data.introvert_count ?? "—";
  if (extEl)   extEl.textContent   = data.extrovert_count ?? "—";
}

// ─── Inicijalizacija pri učitavanju stranice ───────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  if (document.querySelector(".charts-panel")) {
    loadPlotlyCharts();
    // Auto-osvježi svakih 30 sekundi
    setInterval(loadPlotlyCharts, 30_000);
  }
});
