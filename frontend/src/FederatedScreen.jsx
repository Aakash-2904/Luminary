import { useState, useEffect } from "react";

const BASE_URL = "http://localhost:8000";

export default function FederatedScreen() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [animStep, setAnimStep] = useState(0);

  useEffect(() => {
    fetch(`${BASE_URL}/federated/status`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  // Animate node steps
  useEffect(() => {
    if (!data) return;
    const t = setInterval(() => setAnimStep(s => (s + 1) % (data.nodes.length + 2)), 1200);
    return () => clearInterval(t);
  }, [data]);

  const S = {
    page:  { padding:"40px", maxWidth:1000, margin:"0 auto" },
    card:  { background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:12, padding:24, marginBottom:16 },
    tag:   { display:"inline-block", fontSize:11, fontWeight:700, padding:"3px 10px", borderRadius:20, textTransform:"uppercase", letterSpacing:1 },
    green: { background:"rgba(5,150,105,0.15)", color:"#34d399", border:"1px solid rgba(5,150,105,0.3)" },
    purple:{ background:"rgba(124,58,237,0.15)", color:"#a78bfa", border:"1px solid rgba(124,58,237,0.3)" },
  };

  if (loading) return (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"center", height:"calc(100vh - 65px)", color:"#64748b" }}>
      Loading federated network status...
    </div>
  );

  if (!data) return (
    <div style={{ display:"flex", alignItems:"center", justifyContent:"center", height:"calc(100vh - 65px)", color:"#ef4444" }}>
      ⚠️ Backend offline — start uvicorn first
    </div>
  );

  return (
    <div style={S.page}>
      {/* Header */}
      <div style={{ marginBottom:32 }}>
        <div style={{ fontSize:11, fontWeight:700, color:"#7c3aed", textTransform:"uppercase", letterSpacing:2, marginBottom:12 }}>
          Live Network
        </div>
        <h2 style={{ fontSize:32, fontWeight:800, marginBottom:8 }}>Federated Learning Network</h2>
        <p style={{ fontSize:15, color:"#64748b", lineHeight:1.7 }}>
          Each university trains locally on its own data. Only encrypted model weights are shared.
          Raw research data never leaves any university node.
        </p>
      </div>

      {/* Privacy guarantee banner */}
      <div style={{ background:"rgba(5,150,105,0.08)", border:"1px solid rgba(5,150,105,0.2)", borderRadius:12, padding:"16px 24px", marginBottom:24, display:"flex", alignItems:"center", gap:14 }}>
        <span style={{ fontSize:24 }}>🔒</span>
        <div>
          <div style={{ fontSize:14, fontWeight:700, color:"#34d399", marginBottom:4 }}>Privacy Guarantee Active</div>
          <div style={{ fontSize:13, color:"#94a3b8" }}>{data.privacy_guarantee} · Compliance: {data.compliance?.join(" · ")}</div>
        </div>
        <div style={{ marginLeft:"auto", display:"flex", gap:8 }}>
          {data.compliance?.map(c => <span key={c} style={{ ...S.tag, ...S.green }}>{c}</span>)}
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:16, marginBottom:24 }}>
        {[
          ["🏛️", data.global_model.n_nodes, "University Nodes"],
          ["👥", data.global_model.total_researchers, "Researchers Indexed"],
          ["🔐", "0", "Raw Records Shared"],
          ["⚡", "FedAvg", "Aggregation Method"],
        ].map(([icon, val, label]) => (
          <div key={label} style={{ ...S.card, textAlign:"center", padding:20 }}>
            <div style={{ fontSize:24, marginBottom:8 }}>{icon}</div>
            <div style={{ fontSize:22, fontWeight:800, color:"#a78bfa", marginBottom:4 }}>{val}</div>
            <div style={{ fontSize:12, color:"#64748b" }}>{label}</div>
          </div>
        ))}
      </div>

      {/* FL Round Visualization */}
      <div style={S.card}>
        <div style={{ fontSize:16, fontWeight:700, marginBottom:20 }}>Round 1 — Federated Training Flow</div>

        {/* Pipeline steps */}
        <div style={{ display:"flex", alignItems:"center", gap:0, marginBottom:28, overflowX:"auto" }}>
          {["Fetch Data", "Local Training", "Encrypt Weights", "Send to Server", "FedAvg", "Global Model"].map((step, i) => (
            <div key={step} style={{ display:"flex", alignItems:"center", flexShrink:0 }}>
              <div style={{
                padding:"10px 16px", borderRadius:8, fontSize:12, fontWeight:600, textAlign:"center",
                background: animStep > i ? "rgba(124,58,237,0.3)" : "rgba(255,255,255,0.04)",
                border: animStep > i ? "1px solid #7c3aed" : "1px solid rgba(255,255,255,0.08)",
                color: animStep > i ? "#a78bfa" : "#64748b",
                transition:"all .4s ease",
                boxShadow: animStep > i ? "0 0 12px rgba(124,58,237,0.3)" : "none"
              }}>
                {animStep > i ? "✓ " : ""}{step}
              </div>
              {i < 5 && <div style={{ color: animStep > i ? "#7c3aed" : "#4b5563", fontSize:18, margin:"0 6px", transition:"color .4s" }}>→</div>}
            </div>
          ))}
        </div>

        {/* Node cards */}
        <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))", gap:14 }}>
          {data.nodes.map((node, i) => (
            <div key={node.university} style={{
              background:"rgba(255,255,255,0.03)", border:"1px solid rgba(255,255,255,0.07)",
              borderRadius:10, padding:18, transition:"all .4s",
              borderColor: animStep > i ? "rgba(124,58,237,0.4)" : "rgba(255,255,255,0.07)",
              boxShadow: animStep > i ? "0 0 16px rgba(124,58,237,0.1)" : "none"
            }}>
              <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
                <div style={{ width:36, height:36, borderRadius:8, background:"linear-gradient(135deg,#7c3aed,#2563eb)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:16 }}>🏛️</div>
                <div>
                  <div style={{ fontSize:13, fontWeight:700, color:"#e2e8f0" }}>{node.university}</div>
                  <div style={{ fontSize:11, color:"#64748b" }}>{node.researchers_trained} researchers trained</div>
                </div>
              </div>

              <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:12 }}>
                  <span style={{ color:"#64748b" }}>Training Status</span>
                  <span style={{ color:"#34d399", fontWeight:600 }}>{node.status}</span>
                </div>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:12 }}>
                  <span style={{ color:"#64748b" }}>Data Shared</span>
                  <span style={{ color:"#a78bfa", fontWeight:600 }}>{node.data_shared}</span>
                </div>
                <div style={{ display:"flex", justifyContent:"space-between", fontSize:12 }}>
                  <span style={{ color:"#64748b" }}>Raw Data Left Node</span>
                  <span style={{ color:"#ef4444", fontWeight:600 }}>
                    {node.raw_data_left_university ? "⚠️ Yes" : "🔒 Never"}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Encryption explanation */}
      <div style={S.card}>
        <div style={{ fontSize:16, fontWeight:700, marginBottom:16 }}>How Encryption Works</div>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:16 }}>
          {[
            ["1️⃣", "Local Encoding", "Each researcher's abstract is encoded into a 256-dim vector using SciBERT at the university's own server."],
            ["2️⃣", "Node Encryption", "The vector is encrypted with a university-specific key before it leaves the local server. The key never leaves the node."],
            ["3️⃣", "Central Aggregation", "Only encrypted weights reach Luminary's central server. FedAvg aggregates them without ever decrypting raw research content."],
          ].map(([num, title, desc]) => (
            <div key={title} style={{ background:"rgba(255,255,255,0.02)", borderRadius:10, padding:18, border:"1px solid rgba(255,255,255,0.06)" }}>
              <div style={{ fontSize:24, marginBottom:10 }}>{num}</div>
              <div style={{ fontSize:14, fontWeight:700, color:"#c4b5fd", marginBottom:8 }}>{title}</div>
              <div style={{ fontSize:13, color:"#94a3b8", lineHeight:1.6 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}