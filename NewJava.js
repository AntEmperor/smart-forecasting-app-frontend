// ======== CONFIGURE YOUR BACKEND URL HERE ========
// Final working port must be 8001
const API_URL = "http://127.0.0.1:8001";
// =================================================
const API_BASE = "https://emperorgrid.loca.lt";

// Global State and Helpers
const charts = {};
const nowStamp = () => new Date().toLocaleString();
const toNum = (v) => (typeof v === "number" ? v : Number(v || 0));

// --- FEATURE ENGINEERING HELPERS (for seasonal accuracy) ---

function oneHotSeason(season) {
    return {
        season_rainy: season === "Rainy" ? 1 : 0,
        season_dry: season === "Dry" ? 1 : 0,
        season_harmattan: season === "Harmattan" ? 1 : 0,
    };
}

function bandToTemperature(band) {
    if (!band) return 0;
    if (band === "Cool") return 23;      // ≤25°C
    if (band === "Warm") return 29.5;    // 26–35°C midpoint
    if (band === "Hot") return 35.5;     // >35°C nominal
    return Number(band) || 0;
}

// --- UI & CHART HELPERS (UPDATED for Smoothing and Styling) ---

function ensureChart(canvasId, label) {
    const ctx = document.getElementById(canvasId)?.getContext("2d");
    if (!ctx) return null;
    if (charts[canvasId]) return charts[canvasId];

    let initialLabels = [];
    if (canvasId.includes("hourly")) {
        initialLabels = Array.from({ length: 25 }, (_, i) => `${i + 1}`);
    } else if (canvasId.includes("daily")) {
        initialLabels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'];
    } else if (canvasId.includes("weekly")) {
        initialLabels = ['Week 1', 'Week 2', 'Week 3', 'Week 4'];
    }
    
    charts[canvasId] = new Chart(ctx, {
        type: "line",
        data: { labels: initialLabels, datasets: [{ 
            label, 
            data: [], 
            borderWidth: 2, 
            tension: 0.25,
            // ✅ Recommendation (d): Chart Smoothing/Aesthetics
            pointRadius: 3, 
            borderColor: "rgba(54, 162, 235, 0.8)",
            backgroundColor: "rgba(54, 162, 235, 0.2)",
            fill: true
        }] },
        options: { responsive: true, scales: { y: { beginAtZero: true } }, plugins: { legend: { display: true } } }
    });
    return charts[canvasId];
}

// Update the right-hand KPI + chart for a tab
function updateUI(kind, resp) {
    const valueEl = document.getElementById(`${kind}-value`);
    const modelEl = document.getElementById(`${kind}-model`);
    const whenEl  = document.getElementById(`${kind}-when`);
    const statusEl = document.getElementById(`status-${kind}`); // Reference to error status area
    if (!valueEl) return;

    // Clear previous error messages
    if (statusEl) statusEl.textContent = "";

    const predicted = (resp?.prediction_MW ?? resp?.prediction ?? "--");
    valueEl.textContent = (typeof predicted === "number") ? Number(predicted).toFixed(2) : predicted;
    modelEl.textContent = resp?.model_used ? String(resp.model_used).toUpperCase() : "—";
    whenEl.textContent  = nowStamp();

    // Plot a trend array from the backend, or a single point
    const trend = Array.isArray(resp?.trend) ? resp.trend : [Number(predicted) || 0];
    const labels = trend.map((_, i) => `${i + 1}`);

    const chart = ensureChart(`chart-${kind}`, `${kind[0].toUpperCase() + kind.slice(1)} Forecast (MW)`);
    if (chart) {
        chart.data.labels = labels;
        chart.data.datasets[0].data = trend;
        chart.update();
    }
}

// Toggle loading state on button
function setLoading(btn, on = true) {
    if (on) { btn.classList.add("btn-loading"); btn.dataset.orig = btn.innerHTML; btn.innerHTML = "Loading…"; }
    else { btn.classList.remove("btn-loading"); if (btn.dataset.orig) btn.innerHTML = btn.dataset.orig; }
}

// --- API CALL LOGIC (CORRECTED) ---

async function callPredict(model, data) {
    const res = await fetch(`${API_URL}/predict/${model}`, { 
        method: "POST", 
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    if (!res.ok) {
        throw new Error(`API call failed with status: ${res.status}. Message: ${await res.text()}`);
    }
    
    return await res.json();
}

// --- FORM SUBMIT HANDLERS ---

/* HOURLY */
const hourlyForm = document.getElementById("form-hourly");
if (hourlyForm) {
    hourlyForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("hourlySubmit");
        const statusEl = document.getElementById("status-hourly");
        if(statusEl) statusEl.textContent = "";

        const f = Object.fromEntries(new FormData(hourlyForm).entries());

        // ✅ Recommendation (b): Light Validation
        if (!f.hour || !f.temp_level) {
             if(statusEl) { statusEl.textContent = "Error: Please select an hour and temperature band."; statusEl.style.color = "red"; }
             return;
        }

        setLoading(btn, true);

        try {
            const payload = {
                // Primary features
                hour: Number(f.hour),
                day: Number(f.day),
                temperature: bandToTemperature(f.temp_level),
                
                // Historical Lags (Use strong non-zero defaults if form fields are empty/zero)
                last_actual: toNum(f.last_actual) || 6000.0,
                last_pred: toNum(f.last_pred) || 6200.0,

                // Optional/Historical Features with NON-ZERO DEFAULTS
                humidity: toNum(f.humidity) || 75.0,
                max_temp_yesterday: toNum(f.max_temp_yesterday) || 30.0,
                rolling_mean_24h: toNum(f.rolling_mean_24h) || 6050.0,
                rolling_std_24h: toNum(f.rolling_std_24h) || 350.0,
                
                // Seasonal one-hot encoding
                ...oneHotSeason(f.season),
            };

            const resp = await callPredict("hourly", payload);
            updateUI("hourly", resp);
        } catch (err) {
            console.error(err);
            // ✅ Recommendation (c): Graceful Error Feedback
            if(statusEl) { statusEl.textContent = "Error: " + (err.message || "Network issue"); statusEl.style.color = "red"; }
        } finally { setLoading(btn, false); }
    });
}

/* DAILY */
const dailyForm = document.getElementById("form-daily");
if (dailyForm) {
    dailyForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("dailySubmit");
        const statusEl = document.getElementById("status-daily");
        if(statusEl) statusEl.textContent = "";

        const f = Object.fromEntries(new FormData(dailyForm).entries());

        // ✅ Recommendation (b): Light Validation
        if (!f.day || !f.temp_level) {
             if(statusEl) { statusEl.textContent = "Error: Please select a day and temperature band."; statusEl.style.color = "red"; }
             return;
        }

        setLoading(btn, true);

        try {
            // Helper to handle holiday one-hot encoding
            const holiday_type_national = f.daily_holiday_type === 'national' ? 1 : 0;
            const holiday_type_religious = f.daily_holiday_type === 'religious' ? 1 : 0;

            const payload = {
                // Primary features
                day: Number(f.day),
                temperature: bandToTemperature(f.temp_level),
                
                // Historical Lags (Enforce non-zero defaults to prevent flat output)
                last_actual: toNum(f.last_actual) || 580000.0, // Lag 1 day (kW)
                last_pred: toNum(f.last_pred) || 590000.0,    // Lag 7 days (kW)
                
                // Advanced/Categorical features
                is_weekend: toNum(f.is_weekend),
                is_lockdown: toNum(f.is_lockdown),
                holiday_type_national: holiday_type_national,
                holiday_type_religious: holiday_type_religious,

                // Seasonal one-hot encoding
                ...oneHotSeason(f.season),
            };

            const resp = await callPredict("daily", payload);
            updateUI("daily", resp);
        } catch (err) {
            console.error(err);
            // ✅ Recommendation (c): Graceful Error Feedback
            if(statusEl) { statusEl.textContent = "Error: " + (err.message || "Network issue"); statusEl.style.color = "red"; }
        } finally { setLoading(btn, false); }
    });
}

/* WEEKLY */
const weeklyForm = document.getElementById("form-weekly");
if (weeklyForm) {
    weeklyForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("weeklySubmit");
        const statusEl = document.getElementById("status-weekly");
        if(statusEl) statusEl.textContent = "";

        const f = Object.fromEntries(new FormData(weeklyForm).entries());
        
        // Use a date picker field called 'start_date' for the weekly query
        const startDate = f.start_date_utc; // Assuming a field named 'start_date' holds the date

        // ✅ Recommendation (b): Light Validation
        if (!startDate) {
             if(statusEl) { statusEl.textContent = "Error: Please select a valid start date."; statusEl.style.color = "red"; }
             return;
        }

        setLoading(btn, true);

        try {
            // ✅ Recommendation (a): Weekly Payload Simplification
            // The backend only needs the start_date_utc for its internal 7-day loop.
            const payload = {
                start_date_utc: startDate 
            };
            
            // NOTE: If you later upgrade your backend to accept a rich payload for weekly, 
            // you can easily restore the commented-out features below:
            /* const payload = {
                week: Number(f.week) || 40,
                temperature: bandToTemperature(f.temp_level) || 30.0,
                last_actual: toNum(f.last_actual) || 3700000.0,
                last_pred: toNum(f.last_pred) || 3900000.0,
                rolling_mean_3w: toNum(f.rolling_mean_3w) || 3800000.0,
                rolling_std_3w: toNum(f.rolling_std_3w) || 200000.0,
                prev_peak: toNum(f.prev_peak) || 7000.0,
                peak_ratio: toNum(f.peak_ratio) || 1.05,
                ...oneHotSeason(f.season),
            }; 
            */

            const resp = await callPredict("weekly", payload);
            updateUI("weekly", resp);
        } catch (err) {
            console.error(err);
            // ✅ Recommendation (c): Graceful Error Feedback
            if(statusEl) { statusEl.textContent = "Error: " + (err.message || "Network issue"); statusEl.style.color = "red"; }
        } finally { setLoading(btn, false); }
    });
}

/* FEEDBACK */
const feedbackForm = document.getElementById("form-feedback");
if (feedbackForm) {
    feedbackForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fd = new FormData(feedbackForm);
        const body = {
            model: fd.get("model"),
            prediction: toNum(fd.get("prediction")),
            actual: toNum(fd.get("actual")),
            comments: fd.get("comments") || ""
        };
        try {
            const res = await fetch(`${API_URL}/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });
            if (!res.ok) throw new Error(await res.text());
            
            const li = document.createElement("li");
            li.textContent = `[${nowStamp()}] ${body.model.toUpperCase()} — pred: ${body.prediction} MW, actual: ${body.actual} MW — ${body.comments}`;
            document.getElementById("feedback-list").prepend(li);
            feedbackForm.reset();
            alert("Feedback submitted ✅");
        } catch (err) {
            console.error(err);
            alert("Failed to submit feedback.");
        }
    });
}

// --- INITIALIZATION ---

// Populate hourly dropdown (0:00 to 23:00)
document.addEventListener('DOMContentLoaded', () => {
    const hourlySelect = document.getElementById('hourly_hour_select');
    if (hourlySelect) {
        let optionsHtml = '';
        for (let h = 0; h < 24; h++) {
            optionsHtml += `<option value="${h}">${h}:00</option>`;
        }
        hourlySelect.innerHTML = optionsHtml;
    }

    // Initialize empty charts for a nice first render
    ["hourly", "daily", "weekly"].forEach(kind => ensureChart(`chart-${kind}`, `${kind[0].toUpperCase() + kind.slice(1)} Forecast (MW)`));
});

// Download Feedback CSV
document.getElementById("downloadBtn")?.addEventListener("click", () => {
    window.location.href = `${API_URL}/feedback/csv`;
});

// Reset Charts & Backend
document.getElementById("resetBtn")?.addEventListener("click", async () => {
    try {
        // Clear charts
        Object.values(charts).forEach(c => { c.data.labels = []; c.data.datasets[0].data = []; c.update(); });
        // Backend reset (best-effort)
        await fetch(`${API_URL}/reset`, { method: "POST" }).catch(() => { });
        alert("Reset complete (frontend charts and attempted backend reset).");
    } catch { alert("Reset failed."); }
});