import { useState, useEffect, useRef } from "react";

const FEATURES = ["Log_Amount","Hour","V1","V2","V3","V4","V5","Is_Online","Is_Foreign","Transactions_Last1Hr","Suspicious_Hour","Risk_Score"];

function engineerFeatures(t) {
  const log_amount = Math.log1p(t.Amount);
  const suspicious_hour = t.Hour < 6 ? 1 : 0;
  const risk_score = t.Is_Online * 0.3 + t.Is_Foreign * 0.4 + suspicious_hour * 0.3;
  return [log_amount, t.Hour, t.V1, t.V2, t.V3, t.V4, t.V5,
          t.Is_Online, t.Is_Foreign, t.Transactions_Last1Hr, suspicious_hour, risk_score];
}

// Simplified logistic regression weights (pre-computed approximation)
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function predictFraud(txn) {
  const f = engineerFeatures(txn);
  // Weights tuned to match the dataset patterns
  const w = [-0.3, -1.8, -2.1, 1.8, -2.4, 1.9, -1.6, 0.4, 1.2, 1.5, 2.1, 2.8];
  const bias = -2.5;
  const z = w.reduce((s, wi, i) => s + wi * f[i], bias);
  const prob = sigmoid(z);
  return { prob: Math.min(0.999, Math.max(0.001, prob)), features: f };
}

const PRESETS = [
  { label: "Normal Purchase", icon: "🛒", Amount: 45, Hour: 14, V1: 0.1, V2: -0.2, V3: 0.3, V4: -0.1, V5: 0.2, Is_Online: 0, Is_Foreign: 0, Transactions_Last1Hr: 2 },
  { label: "Midnight Fraud", icon: "🌙", Amount: 3200, Hour: 2, V1: -4.1, V2: 3.5, V3: -5.2, V4: 4.0, V5: -3.0, Is_Online: 1, Is_Foreign: 1, Transactions_Last1Hr: 12 },
  { label: "Foreign Online", icon: "🌐", Amount: 180, Hour: 21, V1: -1.5, V2: 1.2, V3: -1.8, V4: 1.0, V5: -0.8, Is_Online: 1, Is_Foreign: 1, Transactions_Last1Hr: 3 },
  { label: "ATM Withdrawal", icon: "🏧", Amount: 200, Hour: 10, V1: 0.5, V2: -0.3, V3: 0.4, V4: -0.2, V5: 0.1, Is_Online: 0, Is_Foreign: 0, Transactions_Last1Hr: 1 },
];

const EMPTY = { Amount: "", Hour: "", V1: "", V2: "", V3: "", V4: "", V5: "", Is_Online: 0, Is_Foreign: 0, Transactions_Last1Hr: "" };

export default function App() {
  const [txn, setTxn] = useState(PRESETS[0]);
  const [result, setResult] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [history, setHistory] = useState([]);
  const [tab, setTab] = useState("analyze");
  const scanRef = useRef(null);

  const handleChange = (k, v) => setTxn(p => ({ ...p, [k]: v }));

  const analyze = () => {
    setScanning(true);
    setResult(null);
    clearTimeout(scanRef.current);
    scanRef.current = setTimeout(() => {
      const r = predictFraud({
        Amount: parseFloat(txn.Amount) || 0,
        Hour: parseInt(txn.Hour) || 0,
        V1: parseFloat(txn.V1) || 0,
        V2: parseFloat(txn.V2) || 0,
        V3: parseFloat(txn.V3) || 0,
        V4: parseFloat(txn.V4) || 0,
        V5: parseFloat(txn.V5) || 0,
        Is_Online: parseInt(txn.Is_Online) || 0,
        Is_Foreign: parseInt(txn.Is_Foreign) || 0,
        Transactions_Last1Hr: parseInt(txn.Transactions_Last1Hr) || 0,
      });
      const entry = {
        id: Date.now(),
        time: new Date().toLocaleTimeString(),
        amount: txn.Amount,
        prob: r.prob,
        isFraud: r.prob > 0.5,
        label: txn.Is_Online ? "Online" : "In-store",
      };
      setHistory(h => [entry, ...h].slice(0, 20));
      setResult(r);
      setScanning(false);
    }, 1400);
  };

  const fraudCount = history.filter(h => h.isFraud).length;
  const legitCount = history.filter(h => !h.isFraud).length;

  const getRisk = (p) => p > 0.7 ? { label: "HIGH RISK", color: "#ff3b3b", bg: "rgba(255,59,59,0.12)" }
    : p > 0.4 ? { label: "MEDIUM RISK", color: "#ff9500", bg: "rgba(255,149,0,0.12)" }
    : { label: "LOW RISK", color: "#30d158", bg: "rgba(48,209,88,0.12)" };

  return (
    <div style={{
      minHeight: "100vh", background: "#0a0a0f",
      fontFamily: "'Courier New', monospace",
      color: "#e0e0e0", padding: "0",
      backgroundImage: "radial-gradient(ellipse at 20% 50%, rgba(0,80,255,0.04) 0%, transparent 60%), radial-gradient(ellipse at 80% 20%, rgba(255,0,80,0.04) 0%, transparent 60%)"
    }}>
      {/* Header */}
      <div style={{ borderBottom: "1px solid rgba(255,255,255,0.07)", padding: "18px 32px", display: "flex", alignItems: "center", justifyContent: "space-between", background: "rgba(255,255,255,0.02)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{ width: 36, height: 36, background: "linear-gradient(135deg,#0050ff,#ff003c)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>💳</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: "bold", letterSpacing: "0.12em", color: "#fff" }}>FRAUD<span style={{ color: "#0050ff" }}>SHIELD</span></div>
            <div style={{ fontSize: 10, color: "#555", letterSpacing: "0.2em" }}>AI DETECTION SYSTEM v2.0</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {["analyze","history","stats"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{
              background: tab === t ? "rgba(0,80,255,0.2)" : "transparent",
              border: `1px solid ${tab === t ? "#0050ff" : "rgba(255,255,255,0.1)"}`,
              color: tab === t ? "#4d8aff" : "#666",
              padding: "6px 16px", borderRadius: 6, cursor: "pointer",
              fontSize: 11, letterSpacing: "0.15em", textTransform: "uppercase"
            }}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 24px" }}>

        {/* ANALYZE TAB */}
        {tab === "analyze" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 380px", gap: 20 }}>
            {/* Left: Form */}
            <div>
              {/* Presets */}
              <div style={{ marginBottom: 20 }}>
                <div style={{ fontSize: 10, letterSpacing: "0.2em", color: "#444", marginBottom: 10 }}>QUICK PRESETS</div>
                <div style={{ display: "flex", gap: 8 }}>
                  {PRESETS.map(p => (
                    <button key={p.label} onClick={() => setTxn(p)} style={{
                      flex: 1, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
                      borderRadius: 8, padding: "10px 6px", cursor: "pointer", color: "#aaa",
                      fontSize: 11, transition: "all 0.2s"
                    }}
                      onMouseEnter={e => e.currentTarget.style.borderColor = "#0050ff"}
                      onMouseLeave={e => e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)"}
                    >
                      <div style={{ fontSize: 18, marginBottom: 4 }}>{p.icon}</div>
                      <div style={{ fontSize: 9, letterSpacing: "0.1em" }}>{p.label.toUpperCase()}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Transaction Details */}
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: 20, marginBottom: 16 }}>
                <div style={{ fontSize: 10, letterSpacing: "0.2em", color: "#444", marginBottom: 16 }}>TRANSACTION DETAILS</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                  {[
                    { key: "Amount", label: "Amount ($)", type: "number", placeholder: "0.00" },
                    { key: "Hour", label: "Hour (0–23)", type: "number", placeholder: "14" },
                    { key: "Transactions_Last1Hr", label: "Tx Last 1Hr", type: "number", placeholder: "2" },
                  ].map(f => (
                    <div key={f.key}>
                      <div style={{ fontSize: 9, color: "#555", letterSpacing: "0.15em", marginBottom: 6 }}>{f.label}</div>
                      <input value={txn[f.key]} onChange={e => handleChange(f.key, e.target.value)}
                        type={f.type} placeholder={f.placeholder}
                        style={{ width: "100%", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 6, padding: "9px 12px", color: "#fff", fontSize: 13, fontFamily: "inherit", boxSizing: "border-box", outline: "none" }} />
                    </div>
                  ))}
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
                  {[
                    { key: "Is_Online", label: "Online Transaction" },
                    { key: "Is_Foreign", label: "Foreign Transaction" },
                  ].map(f => (
                    <div key={f.key} onClick={() => handleChange(f.key, txn[f.key] ? 0 : 1)}
                      style={{ background: txn[f.key] ? "rgba(0,80,255,0.12)" : "rgba(255,255,255,0.03)", border: `1px solid ${txn[f.key] ? "#0050ff" : "rgba(255,255,255,0.08)"}`, borderRadius: 8, padding: "12px 16px", cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "space-between", transition: "all 0.2s" }}>
                      <span style={{ fontSize: 12, color: txn[f.key] ? "#4d8aff" : "#666" }}>{f.label}</span>
                      <div style={{ width: 32, height: 18, background: txn[f.key] ? "#0050ff" : "rgba(255,255,255,0.1)", borderRadius: 9, position: "relative", transition: "all 0.2s" }}>
                        <div style={{ position: "absolute", top: 3, left: txn[f.key] ? 17 : 3, width: 12, height: 12, background: "#fff", borderRadius: "50%", transition: "all 0.2s" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* PCA Features */}
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: 20, marginBottom: 16 }}>
                <div style={{ fontSize: 10, letterSpacing: "0.2em", color: "#444", marginBottom: 16 }}>PCA FEATURES (V1–V5)</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(5,1fr)", gap: 10 }}>
                  {["V1","V2","V3","V4","V5"].map(v => (
                    <div key={v}>
                      <div style={{ fontSize: 9, color: "#555", letterSpacing: "0.15em", marginBottom: 6 }}>{v}</div>
                      <input value={txn[v]} onChange={e => handleChange(v, e.target.value)}
                        type="number" placeholder="0.0"
                        style={{ width: "100%", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 6, padding: "9px 10px", color: "#fff", fontSize: 13, fontFamily: "inherit", boxSizing: "border-box", outline: "none" }} />
                    </div>
                  ))}
                </div>
              </div>

              {/* Analyze Button */}
              <button onClick={analyze} disabled={scanning} style={{
                width: "100%", padding: "16px", background: scanning ? "rgba(0,80,255,0.15)" : "linear-gradient(135deg,#0050ff,#0030cc)",
                border: `1px solid ${scanning ? "#0050ff" : "#0050ff"}`, borderRadius: 10,
                color: "#fff", fontSize: 13, letterSpacing: "0.2em", cursor: scanning ? "not-allowed" : "pointer",
                fontFamily: "inherit", fontWeight: "bold", transition: "all 0.3s",
                boxShadow: scanning ? "none" : "0 0 30px rgba(0,80,255,0.3)"
              }}>
                {scanning ? "⬡ SCANNING TRANSACTION..." : "⬡ ANALYZE TRANSACTION"}
              </button>
            </div>

            {/* Right: Result Panel */}
            <div>
              <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: 24, position: "sticky", top: 20 }}>
                <div style={{ fontSize: 10, letterSpacing: "0.2em", color: "#444", marginBottom: 20 }}>ANALYSIS RESULT</div>

                {!result && !scanning && (
                  <div style={{ textAlign: "center", padding: "40px 0", color: "#333" }}>
                    <div style={{ fontSize: 48, marginBottom: 12 }}>⬡</div>
                    <div style={{ fontSize: 11, letterSpacing: "0.15em" }}>AWAITING INPUT</div>
                  </div>
                )}

                {scanning && (
                  <div style={{ textAlign: "center", padding: "30px 0" }}>
                    <ScannerAnim />
                    <div style={{ fontSize: 11, color: "#0050ff", letterSpacing: "0.2em", marginTop: 16 }}>ANALYZING PATTERN...</div>
                  </div>
                )}

                {result && !scanning && (() => {
                  const risk = getRisk(result.prob);
                  const pct = Math.round(result.prob * 100);
                  return (
                    <div>
                      {/* Verdict */}
                      <div style={{ textAlign: "center", padding: "20px 0", background: risk.bg, borderRadius: 10, marginBottom: 20, border: `1px solid ${risk.color}22` }}>
                        <div style={{ fontSize: 36, marginBottom: 8 }}>{result.prob > 0.5 ? "⚠️" : "✅"}</div>
                        <div style={{ fontSize: 16, fontWeight: "bold", color: risk.color, letterSpacing: "0.1em" }}>
                          {result.prob > 0.5 ? "FRAUD DETECTED" : "LEGITIMATE"}
                        </div>
                        <div style={{ fontSize: 11, color: risk.color, opacity: 0.7, marginTop: 4, letterSpacing: "0.15em" }}>{risk.label}</div>
                      </div>

                      {/* Gauge */}
                      <div style={{ marginBottom: 20 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555", marginBottom: 8 }}>
                          <span>FRAUD PROBABILITY</span><span style={{ color: risk.color }}>{pct}%</span>
                        </div>
                        <div style={{ height: 8, background: "rgba(255,255,255,0.06)", borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${pct}%`, background: `linear-gradient(90deg, #30d158, #ff9500, #ff3b3b)`, borderRadius: 4, transition: "width 0.8s ease" }} />
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "#333", marginTop: 4 }}>
                          <span>0% SAFE</span><span>50%</span><span>100% FRAUD</span>
                        </div>
                      </div>

                      {/* Engineered feature breakdown */}
                      <div style={{ fontSize: 10, letterSpacing: "0.15em", color: "#444", marginBottom: 10 }}>FEATURE SIGNALS</div>
                      {[
                        { label: "Log Amount", val: result.features[0].toFixed(3), icon: "💰" },
                        { label: "Suspicious Hour", val: result.features[10] === 1 ? "YES" : "NO", icon: "🌙", alert: result.features[10] === 1 },
                        { label: "Risk Score", val: result.features[11].toFixed(2), icon: "⚡", alert: result.features[11] > 0.6 },
                        { label: "Tx Frequency", val: result.features[9], icon: "🔁", alert: result.features[9] > 7 },
                      ].map(s => (
                        <div key={s.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 10px", borderRadius: 6, marginBottom: 4, background: s.alert ? "rgba(255,59,59,0.06)" : "rgba(255,255,255,0.02)" }}>
                          <span style={{ fontSize: 11, color: "#666" }}>{s.icon} {s.label}</span>
                          <span style={{ fontSize: 11, color: s.alert ? "#ff3b3b" : "#aaa", fontWeight: s.alert ? "bold" : "normal" }}>{s.val}</span>
                        </div>
                      ))}

                      <div style={{ marginTop: 16, padding: "10px 14px", background: "rgba(255,255,255,0.02)", borderRadius: 8, fontSize: 10, color: "#555", lineHeight: 1.6 }}>
                        {result.prob > 0.7 ? "🔴 Multiple high-risk indicators detected. Recommend blocking this transaction immediately." :
                          result.prob > 0.4 ? "🟡 Some suspicious signals present. Manual review recommended." :
                            "🟢 Transaction profile matches normal spending behavior. No action needed."}
                      </div>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        )}

        {/* HISTORY TAB */}
        {tab === "history" && (
          <div>
            <div style={{ fontSize: 10, letterSpacing: "0.2em", color: "#444", marginBottom: 16 }}>TRANSACTION LOG ({history.length})</div>
            {history.length === 0 && (
              <div style={{ textAlign: "center", padding: 60, color: "#333", fontSize: 12, letterSpacing: "0.15em" }}>NO TRANSACTIONS ANALYZED YET</div>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {history.map((h, i) => {
                const risk = getRisk(h.prob);
                return (
                  <div key={h.id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "14px 18px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderLeft: `3px solid ${risk.color}`, borderRadius: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                      <span style={{ fontSize: 16 }}>{h.isFraud ? "⚠️" : "✅"}</span>
                      <div>
                        <div style={{ fontSize: 12, color: "#ddd" }}>${parseFloat(h.amount || 0).toFixed(2)} — {h.label}</div>
                        <div style={{ fontSize: 10, color: "#444", marginTop: 2 }}>{h.time}</div>
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 13, color: risk.color, fontWeight: "bold" }}>{Math.round(h.prob * 100)}%</div>
                      <div style={{ fontSize: 9, color: "#444", letterSpacing: "0.1em" }}>{risk.label}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* STATS TAB */}
        {tab === "stats" && (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 16 }}>
            {[
              { label: "TOTAL ANALYZED", val: history.length, icon: "📊", color: "#4d8aff" },
              { label: "FRAUD DETECTED", val: fraudCount, icon: "⚠️", color: "#ff3b3b" },
              { label: "LEGITIMATE", val: legitCount, icon: "✅", color: "#30d158" },
              { label: "FRAUD RATE", val: history.length ? `${Math.round(fraudCount/history.length*100)}%` : "—", icon: "📈", color: "#ff9500" },
              { label: "AVG FRAUD PROB", val: history.length ? `${Math.round(history.reduce((s,h)=>s+h.prob,0)/history.length*100)}%` : "—", icon: "🎯", color: "#bf5af2" },
              { label: "HIGH RISK TXN", val: history.filter(h=>h.prob>0.7).length, icon: "🔴", color: "#ff453a" },
            ].map(s => (
              <div key={s.label} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: 24 }}>
                <div style={{ fontSize: 28, marginBottom: 12 }}>{s.icon}</div>
                <div style={{ fontSize: 32, fontWeight: "bold", color: s.color, marginBottom: 4 }}>{s.val}</div>
                <div style={{ fontSize: 10, color: "#444", letterSpacing: "0.15em" }}>{s.label}</div>
              </div>
            ))}

            {history.length > 0 && (
              <div style={{ gridColumn: "1/-1", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 12, padding: 24 }}>
                <div style={{ fontSize: 10, letterSpacing: "0.2em", color: "#444", marginBottom: 16 }}>PROBABILITY DISTRIBUTION</div>
                <div style={{ display: "flex", gap: 3, alignItems: "flex-end", height: 80 }}>
                  {Array.from({ length: 10 }, (_, i) => {
                    const lo = i * 0.1, hi = (i + 1) * 0.1;
                    const cnt = history.filter(h => h.prob >= lo && h.prob < hi).length;
                    const maxCnt = Math.max(...Array.from({length:10},(_,j)=>history.filter(h=>h.prob>=j*0.1&&h.prob<(j+1)*0.1).length), 1);
                    const color = i < 4 ? "#30d158" : i < 7 ? "#ff9500" : "#ff3b3b";
                    return (
                      <div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                        <div style={{ fontSize: 9, color: "#555" }}>{cnt}</div>
                        <div style={{ width: "100%", height: `${(cnt/maxCnt)*60+4}px`, background: color, borderRadius: "3px 3px 0 0", opacity: 0.8 }} />
                        <div style={{ fontSize: 8, color: "#333" }}>{Math.round(lo*100)}%</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ScannerAnim() {
  return (
    <div style={{ position: "relative", width: 80, height: 80, margin: "0 auto" }}>
      {[0,1,2].map(i => (
        <div key={i} style={{
          position: "absolute", inset: `${i*12}px`, borderRadius: "50%",
          border: `1px solid rgba(0,80,255,${0.6-i*0.15})`,
          animation: `ping${i} 1.4s ease-out ${i*0.3}s infinite`,
        }} />
      ))}
      <div style={{ position: "absolute", inset: "28px", background: "#0050ff", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14 }}>💳</div>
      <style>{`
        @keyframes ping0 { 0%{transform:scale(1);opacity:1} 100%{transform:scale(1.5);opacity:0} }
        @keyframes ping1 { 0%{transform:scale(1);opacity:.8} 100%{transform:scale(1.6);opacity:0} }
        @keyframes ping2 { 0%{transform:scale(1);opacity:.6} 100%{transform:scale(1.7);opacity:0} }
      `}</style>
    </div>
  );
}
