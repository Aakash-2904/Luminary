import { useState, useEffect } from "react";
import { searchResearchers } from "./api";

// ── BRAND COLORS — Red · Yellow · White · Blue accent ──
const C = {
  bg:        "#03000a",
  bgCard:    "rgba(20,5,40,0.75)",
  bgCardHov: "rgba(30,8,55,0.9)",
  border:    "rgba(220,80,0,0.25)",
  borderHov: "#ff6a00",
  primary:   "#ff4500",
  primaryDk: "#cc3700",
  blue:      "#1d6fa4",
  blueLight: "#38bdf8",
  amber:     "#ffd700",
  amberDk:   "#ffaa00",
  text:      "#fff8ee",
  textSub:   "#ffcca0",
  textMute:  "#a07050",
  success:   "#10b981",
  warning:   "#ffd700",
  danger:    "#ef4444",
  white:     "#ffffff",
};

// ── DATA ──
const RESEARCHERS = [
  { id:"neu_001", name:"Dr. Priya Sharma", university:"Northeastern University", dept:"Khoury College of Computer Sciences", title:"Federated Learning for Multi-Omics Genomic Analysis", status:"ongoing", methodology:["Federated Learning","Transformer","Deep Learning"], domain:["Genomics","Rare Disease","Multi-Omics"], datasets:["Exome Sequences","RNA-seq Data"], irb:true, stage:"early", abstract:"Investigating federated learning approaches for privacy-preserving analysis of multi-omics genomic datasets to classify rare diseases using transformer architectures.", email:"p.sharma@northeastern.edu", scores:{ overall:88, methodology:85, domain:91, dataset:87, stage:88 }},
  { id:"bu_001",  name:"Prof. James Chen", university:"Boston University", dept:"Department of Computer Science", title:"Federated GNN for Multi-Omics Integration", status:"ongoing", methodology:["Federated Learning","Graph Neural Network","Deep Learning"], domain:["Multi-Omics","Bioinformatics","Genomics"], datasets:["Proteomics Data","Genomic Datasets"], irb:true, stage:"mid", abstract:"Developing federated graph neural networks to integrate multi-omics data across institutions while preserving patient privacy and enabling cross-institutional collaboration.", email:"j.chen@bu.edu", scores:{ overall:91, methodology:88, domain:94, dataset:85, stage:90 }},
  { id:"mit_001", name:"Dr. Aisha Patel", university:"MIT", dept:"CSAIL", title:"Privacy-Preserving Analysis of Rare Genomic Variants", status:"published", methodology:["Differential Privacy","Transformer","NLP"], domain:["Genomics","Rare Disease","Privacy"], datasets:["Anonymized Exome Sequences","Variant Databases"], irb:true, stage:"published", abstract:"Proposed differential privacy mechanisms for large-scale genomic variant analysis, enabling cross-institutional studies without exposing individual patient data.", email:"a.patel@mit.edu", scores:{ overall:79, methodology:76, domain:83, dataset:80, stage:72 }},
  { id:"harvard_001", name:"Prof. Michael Torres", university:"Harvard Medical School", dept:"Department of Biomedical Informatics", title:"IRB Dataset: 12,000 Anonymised Exome Sequences", status:"dataset_available", methodology:["Statistical Analysis","Machine Learning","Bioinformatics"], domain:["Rare Disease","Genomics","Clinical Data"], datasets:["12,000 Anonymized Exome Sequences","Clinical Phenotype Data"], irb:true, stage:"published", abstract:"Curated and IRB-approved dataset of 12,000 anonymized exome sequences from rare disease patients, available for cross-institutional research collaboration.", email:"m.torres@hms.harvard.edu", scores:{ overall:84, methodology:72, domain:79, dataset:100, stage:95 }},
  { id:"tufts_001", name:"Prof. Elena Vasquez", university:"Tufts University", dept:"Department of Computer Science", title:"Distributed Machine Learning for Clinical NLP", status:"ongoing", methodology:["Federated Learning","NLP","BERT"], domain:["Clinical NLP","Healthcare","Privacy"], datasets:["De-identified Clinical Notes","EHR Data"], irb:true, stage:"mid", abstract:"Building distributed NLP models trained across hospital networks to extract clinical insights from patient notes without sharing raw medical records.", email:"e.vasquez@tufts.edu", scores:{ overall:71, methodology:74, domain:68, dataset:72, stage:70 }},
  { id:"bu_002",  name:"Dr. Kevin Walsh", university:"Boston University", dept:"Department of Biomedical Engineering", title:"Cross-Institutional MRI Analysis Using Split Learning", status:"published", methodology:["Split Learning","CNN","Medical Imaging"], domain:["Medical Imaging","Neuroscience","Healthcare"], datasets:["Brain MRI Dataset","fMRI Sequences"], irb:true, stage:"published", abstract:"Proposed split learning architecture for cross-institutional MRI analysis, enabling hospitals to collaboratively train deep learning models on neuroimaging data.", email:"k.walsh@bu.edu", scores:{ overall:63, methodology:60, domain:65, dataset:68, stage:55 }},
  { id:"neu_002", name:"Dr. Rahul Mehta", university:"Northeastern University", dept:"Electrical and Computer Engineering", title:"Quantum Algorithms for Optimization in Drug Discovery", status:"ongoing", methodology:["Quantum Computing","QAOA","Optimization"], domain:["Drug Discovery","Quantum","Bioinformatics"], datasets:["Molecular Structure Databases","Protein Folding Data"], irb:false, stage:"mid", abstract:"Leveraging quantum approximate optimization algorithms to solve combinatorial problems in drug discovery pipelines.", email:"r.mehta@northeastern.edu", scores:{ overall:55, methodology:45, domain:50, dataset:40, stage:75 }},
  { id:"harvard_002", name:"Dr. Lisa Park", university:"Harvard University", dept:"John A. Paulson School of Engineering", title:"Secure Aggregation Protocols for Federated Healthcare Networks", status:"ongoing", methodology:["Federated Learning","Secure Aggregation","Cryptography"], domain:["Healthcare","Privacy","Security"], datasets:["Synthetic Patient Records","Federated Benchmark Datasets"], irb:true, stage:"mid", abstract:"Designing cryptographically secure aggregation protocols for federated learning in healthcare settings.", email:"l.park@harvard.edu", scores:{ overall:76, methodology:80, domain:74, dataset:78, stage:72 }},
];

const SUGGESTIONS = ["Federated Learning for Genomics","Medical Imaging Privacy","Quantum Optimization","Clinical NLP","Rare Disease Datasets"];
const UNIVERSITIES = ["All Universities","Northeastern University","Boston University","MIT","Harvard Medical School","Tufts University"];
const NAV_ITEMS = ["Search","Results","Upload","Network"];

// ── HELPERS ──
const scoreColor = s => s >= 80 ? C.success : s >= 65 ? C.warning : C.danger;
const statusLabel = s => ({ ongoing:"Ongoing", published:"Published", dataset_available:"Dataset Available" }[s] || s);
const statusColor = s => ({ ongoing:C.success, published:C.blueLight, dataset_available:C.primary }[s] || C.textSub);
const initials = n => n.split(" ").filter(Boolean).map(w=>w[0]).join("").slice(0,2).toUpperCase();

// ── GLOBAL STYLES ──
const S = {
  app:      { minHeight:"100vh", width:"100vw", background:C.bg, color:C.text, fontFamily:"'Inter','Segoe UI',sans-serif", overflowX:"hidden" },
  nav:      { display:"flex", alignItems:"center", justifyContent:"space-between", padding:"14px 48px", background:"rgba(3,0,10,0.7)", borderBottom:`1px solid rgba(255,69,0,0.2)`, position:"sticky", top:0, zIndex:100, backdropFilter:"blur(20px)" },
  logo:     { fontSize:"22px", fontWeight:800, letterSpacing:"-0.5px", display:"flex", alignItems:"center", gap:8 },
  logoMark: { width:32, height:32, borderRadius:8, background:`linear-gradient(135deg,${C.primary},${C.amber})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:16, fontWeight:900, color:"#fff", boxShadow:`0 0 16px ${C.primary}66` },
  navLink:  { fontSize:"13px", color:C.textSub, cursor:"pointer", padding:"6px 14px", borderRadius:6, transition:"all .2s", fontWeight:500, userSelect:"none" },
  navActive:{ fontSize:"13px", color:C.white, cursor:"pointer", background:`${C.primary}22`, border:`1px solid ${C.primary}44`, padding:"6px 14px", borderRadius:6, fontWeight:600, userSelect:"none" },
  avatar:   { width:34, height:34, borderRadius:10, background:`linear-gradient(135deg,${C.primary},${C.blue})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:13, fontWeight:700, color:"#fff" },
  card:     { background:C.bgCard, border:`1px solid ${C.border}`, borderRadius:14, padding:24 },
  btn:      { background:`linear-gradient(135deg,${C.primary},${C.amber})`, color:"#000", border:"none", padding:"11px 26px", borderRadius:10, fontSize:14, fontWeight:700, cursor:"pointer", transition:"all .2s", letterSpacing:"0.3px" },
  btnOutline:{ background:"transparent", color:C.amber, border:`1px solid ${C.amber}`, padding:"9px 20px", borderRadius:10, fontSize:13, fontWeight:600, cursor:"pointer", transition:"all .2s" },
  btnAmber: { background:`linear-gradient(135deg,${C.amber},${C.amberDk})`, color:"#000", border:"none", padding:"11px 26px", borderRadius:10, fontSize:14, fontWeight:700, cursor:"pointer", transition:"all .2s" },
  pill:     { display:"inline-block", background:`${C.primary}18`, border:`1px solid ${C.primary}33`, color:C.primary, fontSize:12, padding:"3px 10px", borderRadius:20, marginRight:6, marginBottom:6 },
  pillBlue: { display:"inline-block", background:`${C.blueLight}18`, border:`1px solid ${C.blueLight}33`, color:C.blueLight, fontSize:12, padding:"3px 10px", borderRadius:20, marginRight:6, marginBottom:6 },
  tag:      { display:"inline-block", fontSize:11, fontWeight:700, padding:"3px 10px", borderRadius:20, textTransform:"uppercase", letterSpacing:"0.5px" },
  input:    { background:`${C.bgCard}`, border:`1px solid ${C.border}`, borderRadius:10, padding:"11px 16px", color:C.text, fontSize:14, outline:"none", fontFamily:"inherit", transition:"border .2s" },
  divider:  { height:1, background:`${C.border}`, margin:"20px 0" },
};

// ── COSMIC BACKGROUND ──
function CosmicBg() {
  useEffect(() => {
    const canvas = document.getElementById("luminary-bg");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W = canvas.width = window.innerWidth;
    let H = canvas.height = window.innerHeight;
    const resize = () => { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; };
    window.addEventListener("resize", resize);

    // Stars — warp speed streaks
    const N = 280;
    const stars = Array.from({ length: N }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      z: Math.random() * W,
      pz: 0,
    }));

    // Quantum particles
    const QN = 60;
    const qp = Array.from({ length: QN }, () => ({
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 2.5 + 0.5,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      phase: Math.random() * Math.PI * 2,
      col: Math.random() > 0.5 ? "#ff4500" : "#ffd700",
      opacity: Math.random() * 0.6 + 0.2,
    }));

    let frame = 0;
    let raf;

    const draw = () => {
      ctx.fillStyle = "rgba(3,0,10,0.25)";
      ctx.fillRect(0, 0, W, H);
      frame++;

      // Warp star streaks
      const cx = W / 2, cy = H / 2;
      stars.forEach(s => {
        s.pz = s.z;
        s.z -= 5;
        if (s.z <= 0) { s.x = Math.random() * W; s.y = Math.random() * H; s.z = W; s.pz = W; }
        const sx = (s.x - cx) / s.z * W + cx;
        const sy = (s.y - cy) / s.z * W + cy;
        const px = (s.x - cx) / s.pz * W + cx;
        const py = (s.y - cy) / s.pz * W + cy;
        const size = (1 - s.z / W) * 2.5;
        const speed = 1 - s.z / W;
        // Color: white core, amber/red tail
        const r = Math.floor(255);
        const g = Math.floor(180 * speed);
        const b = Math.floor(30 * speed);
        ctx.strokeStyle = `rgba(${r},${g},${b},${speed * 0.9})`;
        ctx.lineWidth = size;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(sx, sy);
        ctx.stroke();
      });

      // Quantum superposition particles — pulsing orbs with entanglement lines
      qp.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;
        p.phase += 0.025;
        if (p.x < 0 || p.x > W) p.vx *= -1;
        if (p.y < 0 || p.y > H) p.vy *= -1;
        const pulse = Math.sin(p.phase) * 0.4 + 0.6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * (1 + pulse * 0.8), 0, Math.PI * 2);
        ctx.fillStyle = p.col === "#ff4500"
          ? `rgba(255,69,0,${p.opacity * pulse})`
          : `rgba(255,215,0,${p.opacity * pulse})`;
        ctx.fill();

        // Entanglement lines to nearby particles
        for (let j = i + 1; j < Math.min(i + 4, QN); j++) {
          const q = qp[j];
          const dx = q.x - p.x, dy = q.y - p.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 140) {
            const alpha = (1 - dist / 140) * 0.18 * pulse;
            ctx.strokeStyle = `rgba(255,150,0,${alpha})`;
            ctx.lineWidth = 0.6;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
          }
        }
      });

      // Central warp glow — red/yellow core
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.min(W, H) * 0.38);
      grd.addColorStop(0, `rgba(255,100,0,${0.06 + Math.sin(frame * 0.012) * 0.02})`);
      grd.addColorStop(0.5, `rgba(255,50,0,${0.03})`);
      grd.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, W, H);

      raf = requestAnimationFrame(draw);
    };

    draw();
    return () => { window.removeEventListener("resize", resize); cancelAnimationFrame(raf); };
  }, []);

  return (
    <canvas id="luminary-bg" style={{
      position:"fixed", top:0, left:0, width:"100vw", height:"100vh",
      zIndex:0, pointerEvents:"none", display:"block"
    }} />
  );
}

// ── ANIMATED PROGRESS BAR ──
function ProgressBar({ value, color, delay=0 }) {
  const [w, setW] = useState(0);
  useEffect(() => { const t = setTimeout(() => setW(value), delay); return () => clearTimeout(t); }, [value, delay]);
  return (
    <div style={{ background:"rgba(255,255,255,0.06)", borderRadius:8, height:7, overflow:"hidden" }}>
      <div style={{ width:`${w}%`, height:"100%", background:color, borderRadius:8, transition:"width 1.1s ease", boxShadow:`0 0 8px ${color}55` }} />
    </div>
  );
}

// ── CIRCULAR SCORE ──
function CircleScore({ score, size=90 }) {
  const [s, setS] = useState(0);
  useEffect(() => { const t = setTimeout(() => setS(score), 300); return () => clearTimeout(t); }, [score]);
  const r = size/2-8, circ = 2*Math.PI*r, dash = circ-(s/100)*circ;
  return (
    <div style={{ position:"relative", width:size, height:size, display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
      <svg width={size} height={size} style={{ position:"absolute", top:0, left:0, transform:"rotate(-90deg)" }}>
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={7} />
        <circle cx={size/2} cy={size/2} r={r} fill="none" stroke={scoreColor(score)} strokeWidth={7}
          strokeDasharray={circ} strokeDashoffset={dash} strokeLinecap="round"
          style={{ transition:"stroke-dashoffset 1.3s ease", filter:`drop-shadow(0 0 5px ${scoreColor(score)})` }} />
      </svg>
      <div style={{ textAlign:"center", zIndex:1 }}>
        <div style={{ fontSize:size>80?19:14, fontWeight:800, color:scoreColor(score) }}>{s}%</div>
        <div style={{ fontSize:9, color:C.textMute, marginTop:1 }}>Collab</div>
      </div>
    </div>
  );
}

// ── TOGGLE ──
function Toggle({ value, onChange, label }) {
  return (
    <div onClick={() => onChange(!value)}
      style={{ display:"flex", alignItems:"center", gap:10, cursor:"pointer", padding:"8px 12px", background:value?`${C.primary}15`:"rgba(255,255,255,0.03)", borderRadius:8, border:`1px solid ${value?C.primary+"44":"rgba(255,255,255,0.06)"}`, transition:"all .2s" }}>
      <div style={{ width:38, height:21, borderRadius:11, background:value?C.primary:"rgba(255,255,255,0.1)", position:"relative", transition:"background .25s", flexShrink:0 }}>
        <div style={{ position:"absolute", top:3, left:value?19:3, width:15, height:15, borderRadius:"50%", background:"#fff", transition:"left .25s", boxShadow:"0 1px 4px rgba(0,0,0,0.3)" }} />
      </div>
      <span style={{ fontSize:12, color:value?C.primary:C.textSub, fontWeight:value?600:400, transition:"color .2s" }}>{label}</span>
    </div>
  );
}

// ── BADGE ──
function Badge({ label, color }) {
  return <span style={{ ...S.tag, background:`${color}18`, color:color, border:`1px solid ${color}33` }}>{label}</span>;
}

// ── TOAST ──
function Toast({ msg, type="success", onDone }) {
  useEffect(() => { const t = setTimeout(onDone, 3500); return () => clearTimeout(t); });
  const color = type==="success" ? C.success : type==="warning" ? C.warning : C.danger;
  return (
    <div style={{ position:"fixed", bottom:32, right:32, background:C.bgCard, border:`1px solid ${color}44`, borderRadius:12, padding:"14px 22px", color:color, fontWeight:600, fontSize:14, zIndex:999, boxShadow:"0 8px 32px rgba(0,0,0,0.5)", display:"flex", alignItems:"center", gap:10, animation:"slideUp .3s ease" }}>
      <span>{type==="success"?"✅":type==="warning"?"⚠️":"❌"}</span> {msg}
    </div>
  );
}

// ── SPINNER ──
function Spinner({ text="Running RAG + QAOA pipeline..." }) {
  return (
    <div style={{ display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:80, gap:16 }}>
      <div style={{ width:44, height:44, border:`3px solid ${C.border}`, borderTop:`3px solid ${C.primary}`, borderRadius:"50%", animation:"spin 0.8s linear infinite" }} />
      <div style={{ color:C.textSub, fontSize:14 }}>{text}</div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
}

// ── SECTION HEADER ──
function SectionHeader({ tag, title, sub }) {
  return (
    <div style={{ marginBottom:28 }}>
      <div style={{ fontSize:11, fontWeight:700, color:C.primary, textTransform:"uppercase", letterSpacing:3, marginBottom:10 }}>{tag}</div>
      <h2 style={{ fontSize:32, fontWeight:800, color:C.white, marginBottom:8, letterSpacing:"-0.5px" }}>{title}</h2>
      {sub && <p style={{ fontSize:15, color:C.textSub, lineHeight:1.7, maxWidth:600 }}>{sub}</p>}
    </div>
  );
}

// ════════════════════════════════════════
// SCREEN 1 — SEARCH
// ════════════════════════════════════════
function SearchScreen({ onSearch, onNav }) {
  const [q, setQ] = useState("");
  return (
    <div style={{ minHeight:"calc(100vh - 65px)", display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:"48px 24px", position:"relative", overflow:"hidden" }}>
      {/* Background glow */}
      <div style={{ position:"absolute", top:"20%", left:"50%", transform:"translateX(-50%)", width:600, height:600, background:`radial-gradient(circle,${C.primary}18 0%,transparent 70%)`, pointerEvents:"none" }} />

      <div style={{ fontSize:11, fontWeight:700, color:C.primary, textTransform:"uppercase", letterSpacing:3, marginBottom:16 }}>Protothon 8.0 — Team Quaffle</div>

      <h1 style={{ fontSize:56, fontWeight:900, textAlign:"center", lineHeight:1.05, marginBottom:16, maxWidth:720, letterSpacing:"-1.5px" }}>
        <span style={{ color:C.white }}>Discover Research.</span><br/>
        <span style={{ background:`linear-gradient(90deg,${C.primary},${C.amber})`, WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Find Collaborators.</span>
      </h1>
      <p style={{ fontSize:17, color:C.textSub, textAlign:"center", marginBottom:40, maxWidth:520, lineHeight:1.8 }}>
        Search across every connected university — in-progress and published work, federated and privacy-preserving.
      </p>

      {/* Search Box */}
      <div style={{ width:"100%", maxWidth:720, background:C.bgCard, border:`1px solid ${C.border}`, borderRadius:18, padding:"22px 26px", boxShadow:`0 0 60px ${C.primary}12, 0 24px 48px rgba(0,0,0,0.3)`, position:"relative", zIndex:1 }}>
        <textarea rows={3} value={q} onChange={e => setQ(e.target.value)}
          placeholder="Describe your research... e.g. federated learning applied to genomic analysis for rare disease classification"
          style={{ background:"transparent", border:"none", outline:"none", color:C.text, fontSize:15, lineHeight:1.7, resize:"none", fontFamily:"inherit", width:"100%", caretColor:C.primary }}
          onKeyDown={e => { if (e.key==="Enter"&&!e.shiftKey){e.preventDefault(); q.trim()&&onSearch(q.trim()); }}}
        />
        <div style={{ height:1, background:C.border, margin:"12px 0" }} />
        <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:10 }}>
          <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
            {SUGGESTIONS.map(s => (
              <span key={s} onClick={() => setQ(s)} style={{ ...S.pill, cursor:"pointer", transition:"all .2s" }}
                onMouseEnter={e => { e.currentTarget.style.background=`${C.primary}35`; e.currentTarget.style.borderColor=C.primary; }}
                onMouseLeave={e => { e.currentTarget.style.background=`${C.primary}18`; e.currentTarget.style.borderColor=`${C.primary}33`; }}
              >{s}</span>
            ))}
          </div>
          <button style={S.btn} onClick={() => q.trim()&&onSearch(q.trim())}
            onMouseEnter={e => e.currentTarget.style.opacity="0.85"}
            onMouseLeave={e => e.currentTarget.style.opacity="1"}
          >Search Luminary →</button>
        </div>
      </div>

      {/* Stats */}
      <div style={{ display:"flex", gap:16, marginTop:40, flexWrap:"wrap", justifyContent:"center", position:"relative", zIndex:1 }}>
        {[["🏛", "10","Researchers indexed"],["🌐","5","Connected universities"],["⚛️","QAOA","Quantum collaboration scoring"],["🔒","FL","Privacy by architecture"]].map(([icon,n,l]) => (
          <div key={n} style={{ ...S.card, textAlign:"center", minWidth:150, padding:"18px 22px", borderColor:C.border }}>
            <div style={{ fontSize:22, marginBottom:6 }}>{icon}</div>
            <div style={{ fontSize:20, fontWeight:800, color:C.primary }}>{n}</div>
            <div style={{ fontSize:12, color:C.textSub, marginTop:4 }}>{l}</div>
          </div>
        ))}
      </div>

      {/* Quick nav to upload */}
      <div style={{ marginTop:32, display:"flex", alignItems:"center", gap:12 }}>
        <span style={{ fontSize:13, color:C.textMute }}>Are you a university?</span>
        <button style={S.btnOutline} onClick={() => onNav("upload")}>Submit Your Research Data →</button>
      </div>
    </div>
  );
}

// ════════════════════════════════════════
// SCREEN 2 — RESULTS
// ════════════════════════════════════════
function ResultsScreen({ query, onBack, onSelect }) {
  const [q, setQ] = useState(query);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uniFilter, setUniFilter] = useState("All Universities");
  const [irbOnly, setIrbOnly] = useState(false);
  const [statusFilter, setStatusFilter] = useState("all");
  const [ongoingOnly, setOngoingOnly] = useState(false);

  useEffect(() => {
    const run = async () => {
      setLoading(true);
      const data = await searchResearchers({ query:q, universityFilter:uniFilter, irbFilter:irbOnly, statusFilter });
      if (data) {
        setResults(data.map(r => ({
          ...r, irb:r.irb_status==="approved",
          scores:{
            overall: r.match_percent||Math.round((r.collaboration_probability||0.7)*100),
            methodology: Math.round((r.breakdown?.methodology_overlap||0.75)*100),
            domain:      Math.round((r.breakdown?.domain_proximity||0.75)*100),
            dataset:     Math.round((r.breakdown?.dataset_compatibility||0.75)*100),
            stage:       Math.round((r.breakdown?.stage_complementarity||0.75)*100),
          }
        })));
      } else {
        setResults(RESEARCHERS.sort((a,b)=>b.scores.overall-a.scores.overall));
      }
      setLoading(false);
    };
    run();
  }, [q, uniFilter, irbOnly, statusFilter]);

  const displayed = results
    .filter(r => uniFilter==="All Universities"||r.university===uniFilter)
    .filter(r => !irbOnly||r.irb)
    .filter(r => statusFilter==="all"||r.status===statusFilter)
    .filter(r => !ongoingOnly||r.status==="ongoing");

  return (
    <div style={{ display:"flex", minHeight:"calc(100vh - 65px)" }}>
      {/* Sidebar */}
      <div style={{ width:230, padding:"24px 18px", borderRight:`1px solid ${C.border}`, flexShrink:0, background:`${C.bgCard}88` }}>
        <div style={{ fontSize:11, fontWeight:700, color:C.primary, marginBottom:18, textTransform:"uppercase", letterSpacing:2 }}>Filters</div>

        <div style={{ marginBottom:22 }}>
          <div style={{ fontSize:11, color:C.textMute, marginBottom:10, fontWeight:700, textTransform:"uppercase", letterSpacing:1 }}>University</div>
          {UNIVERSITIES.map(u => (
            <div key={u} onClick={() => setUniFilter(u)}
              style={{ fontSize:13, padding:"7px 10px", borderRadius:7, marginBottom:3, cursor:"pointer", color:uniFilter===u?C.primary:C.textSub, background:uniFilter===u?`${C.primary}15`:"transparent", borderLeft:uniFilter===u?`2px solid ${C.primary}`:"2px solid transparent", transition:"all .15s", fontWeight:uniFilter===u?600:400 }}>
              {u}
            </div>
          ))}
        </div>

        <div style={{ marginBottom:22 }}>
          <div style={{ fontSize:11, color:C.textMute, marginBottom:10, fontWeight:700, textTransform:"uppercase", letterSpacing:1 }}>Status</div>
          {[["all","All Results"],["ongoing","Ongoing"],["published","Published"],["dataset_available","Dataset Available"]].map(([v,l]) => (
            <div key={v} onClick={() => setStatusFilter(v)}
              style={{ fontSize:13, padding:"7px 10px", borderRadius:7, marginBottom:3, cursor:"pointer", color:statusFilter===v?C.primary:C.textSub, background:statusFilter===v?`${C.primary}15`:"transparent", borderLeft:statusFilter===v?`2px solid ${C.primary}`:"2px solid transparent", transition:"all .15s", fontWeight:statusFilter===v?600:400 }}>
              {l}
            </div>
          ))}
        </div>

        <div style={{ display:"flex", flexDirection:"column", gap:8 }}>
          <Toggle value={irbOnly} onChange={setIrbOnly} label="IRB Approved Only" />
          <Toggle value={ongoingOnly} onChange={setOngoingOnly} label="Ongoing Only" />
        </div>
      </div>

      {/* Main */}
      <div style={{ flex:1, padding:"28px 36px", overflowY:"auto" }}>
        <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom:22, flexWrap:"wrap" }}>
          <button style={{ ...S.btnOutline, padding:"8px 16px", fontSize:13 }} onClick={onBack}>← Back</button>
          <input value={q} onChange={e => setQ(e.target.value)}
            style={{ ...S.input, flex:1, minWidth:200 }}
            onFocus={e => e.target.style.borderColor=C.primary}
            onBlur={e => e.target.style.borderColor=C.border}
            onKeyDown={e => e.key==="Enter"&&setQ(e.target.value)}
          />
        </div>

        {loading ? <Spinner /> : (
          <>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", marginBottom:22 }}>
              <div style={{ fontSize:14, color:C.textSub }}>
                Showing <strong style={{ color:C.primary }}>{displayed.length}</strong> results for <strong style={{ color:C.white }}>"{q}"</strong>
                <span style={{ marginLeft:10, fontSize:12, color:C.textMute, background:`${C.primary}12`, padding:"2px 10px", borderRadius:20, border:`1px solid ${C.primary}22` }}>⚛ QAOA Ranked</span>
              </div>
            </div>

            {displayed.map(r => (
              <div key={r.id}
                style={{ ...S.card, marginBottom:14, display:"flex", gap:22, alignItems:"center", transition:"all .2s", cursor:"pointer" }}
                onMouseEnter={e => { e.currentTarget.style.borderColor=C.borderHov; e.currentTarget.style.boxShadow=`0 0 24px ${C.primary}12`; e.currentTarget.style.background=C.bgCardHov; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor=C.border; e.currentTarget.style.boxShadow="none"; e.currentTarget.style.background=C.bgCard; }}
              >
                <div style={{ flex:1 }}>
                  <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:6 }}>
                    <span style={{ fontSize:11, color:C.primary, fontWeight:700, background:`${C.primary}15`, padding:"2px 10px", borderRadius:20, border:`1px solid ${C.primary}22` }}>{r.university}</span>
                    <Badge label={statusLabel(r.status)} color={statusColor(r.status)} />
                    {r.irb && <Badge label="IRB ✓" color={C.success} />}
                  </div>
                  <div style={{ fontSize:18, fontWeight:700, color:C.white, marginBottom:4 }}>{r.name}</div>
                  <div style={{ fontSize:13, color:C.textSub, marginBottom:8 }}>{r.dept}</div>
                  <div style={{ fontSize:14, color:C.primary, fontWeight:600, marginBottom:8 }}>{r.title}</div>
                  <div style={{ fontSize:13, color:C.textSub, marginBottom:10, lineHeight:1.6 }}>{r.abstract.slice(0,130)}...</div>
                  <div>{r.methodology.slice(0,3).map(m => <span key={m} style={S.pill}>{m}</span>)}</div>
                </div>
                <div style={{ display:"flex", flexDirection:"column", alignItems:"center", gap:10, flexShrink:0 }}>
                  <CircleScore score={r.scores.overall} size={88} />
                  <button style={{ ...S.btn, padding:"8px 18px", fontSize:12 }} onClick={() => onSelect(r)}>View Profile →</button>
                </div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}

// ════════════════════════════════════════
// SCREEN 3 — PROFILE
// ════════════════════════════════════════
function ProfileScreen({ researcher:r, onBack }) {
  const [msg, setMsg] = useState("");
  const [toast, setToast] = useState(false);
  const [sent, setSent] = useState(false);

  const quickMsgs = [
    "Hi! I'd love to explore sharing dataset access and running a joint analysis on our overlapping research areas.",
    "I'm interested in co-authoring a paper that combines our methodologies — I think our work is highly complementary.",
    "I'd appreciate a 30-minute consultation on your federated learning approach for my current project.",
    "Let's explore a joint grant application to NSF/NIH — our research overlap makes a compelling case.",
  ];

  const bars = [
    { label:"Methodology Overlap",   value:r.scores.methodology, color:C.primary },
    { label:"Domain Proximity",      value:r.scores.domain,      color:C.blueLight },
    { label:"Dataset Compatibility", value:r.scores.dataset,     color:C.success },
    { label:"Research Stage Fit",    value:r.scores.stage,       color:C.amber },
  ];

  const insights = [
    { ok:true,  text:`Strong methodology alignment — both use ${r.methodology[0]}` },
    { ok:true,  text:r.irb?"IRB-approved datasets are compatible for cross-institutional use":"Dataset access requires IRB coordination" },
    { ok:true,  text:"Research stages are complementary — collaboration adds clear value" },
    { ok:false, text:"High domain overlap — define collaboration scope clearly to avoid duplication" },
  ];

  return (
    <div style={{ padding:"32px 48px", maxWidth:1200, margin:"0 auto" }}>
      {toast && <Toast msg={`Request sent to ${r.name}!`} type="success" onDone={() => setToast(false)} />}

      <button style={{ ...S.btnOutline, marginBottom:24 }} onClick={onBack}>← Back to Results</button>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1.5fr", gap:24 }}>
        {/* LEFT */}
        <div>
          <div style={{ ...S.card, marginBottom:16 }}>
            <div style={{ display:"flex", gap:16, alignItems:"center", marginBottom:20 }}>
              <div style={{ width:60, height:60, borderRadius:14, background:`linear-gradient(135deg,${C.primary},${C.blue})`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:20, fontWeight:800, color:"#fff", flexShrink:0 }}>
                {initials(r.name)}
              </div>
              <div>
                <div style={{ fontSize:19, fontWeight:800, color:C.white }}>{r.name}</div>
                <div style={{ fontSize:13, color:C.primary, fontWeight:600 }}>{r.university}</div>
                <div style={{ fontSize:12, color:C.textSub }}>{r.dept}</div>
              </div>
            </div>

            <div style={{ display:"flex", gap:6, marginBottom:16, flexWrap:"wrap" }}>
              <Badge label={statusLabel(r.status)} color={statusColor(r.status)} />
              {r.irb && <Badge label="IRB Approved" color={C.success} />}
              <Badge label={`Stage: ${r.stage}`} color={C.blueLight} />
            </div>

            <div style={{ fontSize:14, color:C.primary, fontWeight:600, marginBottom:10, lineHeight:1.5 }}>{r.title}</div>
            <p style={{ fontSize:13, color:C.textSub, lineHeight:1.7, marginBottom:16 }}>{r.abstract}</p>

            <div style={S.divider} />

            <div style={{ marginBottom:14 }}>
              <div style={{ fontSize:11, color:C.textMute, fontWeight:700, textTransform:"uppercase", letterSpacing:1, marginBottom:8 }}>Methodology</div>
              <div>{r.methodology.map(m => <span key={m} style={S.pill}>{m}</span>)}</div>
            </div>
            <div style={{ marginBottom:14 }}>
              <div style={{ fontSize:11, color:C.textMute, fontWeight:700, textTransform:"uppercase", letterSpacing:1, marginBottom:8 }}>Domains</div>
              <div>{r.domain.map(d => <span key={d} style={S.pillBlue}>{d}</span>)}</div>
            </div>
            <div>
              <div style={{ fontSize:11, color:C.textMute, fontWeight:700, textTransform:"uppercase", letterSpacing:1, marginBottom:8 }}>Datasets</div>
              {r.datasets.map(d => (
                <div key={d} style={{ fontSize:13, color:C.textSub, padding:"7px 0", borderBottom:`1px solid ${C.border}`, display:"flex", alignItems:"center", gap:8 }}>
                  <span>{r.irb?"🔓":"🔒"}</span>{d}
                  {r.irb && <span style={{ fontSize:11, color:C.success, marginLeft:"auto" }}>Requestable</span>}
                </div>
              ))}
            </div>
          </div>

          <div style={{ ...S.card, fontSize:13 }}>
            <div style={{ fontSize:11, fontWeight:700, textTransform:"uppercase", letterSpacing:1, marginBottom:8, color:C.textMute }}>Contact</div>
            {sent
              ? <span style={{ color:C.success, fontWeight:600 }}>✅ Request sent — awaiting response</span>
              : <span style={{ color:C.textMute, fontStyle:"italic" }}>🔒 Email revealed after sending request</span>}
          </div>
        </div>

        {/* RIGHT */}
        <div>
          <div style={{ ...S.card, marginBottom:16 }}>
            <div style={{ fontSize:16, fontWeight:700, color:C.white, marginBottom:20 }}>Collaboration Score Breakdown</div>
            <div style={{ display:"flex", alignItems:"center", gap:28, marginBottom:24 }}>
              <CircleScore score={r.scores.overall} size={110} />
              <div style={{ flex:1, fontSize:13, color:C.textSub, lineHeight:1.8 }}>
                Luminary's QAOA quantum optimization scored this researcher across <strong style={{ color:C.white }}>4 collaboration variables</strong> simultaneously. The score reflects complementarity — not just similarity.
              </div>
            </div>
            {bars.map((b,i) => (
              <div key={b.label} style={{ marginBottom:16 }}>
                <div style={{ display:"flex", justifyContent:"space-between", marginBottom:6 }}>
                  <span style={{ fontSize:13, color:C.textSub }}>{b.label}</span>
                  <span style={{ fontSize:13, fontWeight:700, color:b.color }}>{b.value}%</span>
                </div>
                <ProgressBar value={b.value} color={b.color} delay={i*150} />
              </div>
            ))}
            <div style={{ marginTop:20, display:"flex", flexDirection:"column", gap:8 }}>
              {insights.map((ins,i) => (
                <div key={i} style={{ display:"flex", gap:10, alignItems:"flex-start", padding:"8px 12px", background:"rgba(255,255,255,0.02)", borderRadius:8, border:`1px solid ${C.border}` }}>
                  <span style={{ flexShrink:0 }}>{ins.ok?"✅":"⚠️"}</span>
                  <span style={{ fontSize:13, color:ins.ok?C.textSub:C.amber, lineHeight:1.5 }}>{ins.text}</span>
                </div>
              ))}
            </div>
          </div>

          <div style={S.card}>
            <div style={{ fontSize:16, fontWeight:700, color:C.white, marginBottom:14 }}>Send Collaboration Request</div>
            <div style={{ display:"flex", gap:8, flexWrap:"wrap", marginBottom:14 }}>
              {["Share Dataset","Co-author Paper","Method Consult","Joint Grant"].map((q,i) => (
                <button key={q} onClick={() => setMsg(quickMsgs[i])}
                  style={{ ...S.btnOutline, fontSize:12, padding:"6px 12px" }}
                  onMouseEnter={e => { e.currentTarget.style.background=`${C.primary}18`; }}
                  onMouseLeave={e => { e.currentTarget.style.background="transparent"; }}
                >{q}</button>
              ))}
            </div>
            <textarea rows={4} value={msg} onChange={e => setMsg(e.target.value)}
              placeholder="Describe your collaboration interest and what you bring to the table..."
              style={{ ...S.input, width:"100%", resize:"none", marginBottom:12, display:"block" }}
              onFocus={e => e.target.style.borderColor=C.primary}
              onBlur={e => e.target.style.borderColor=C.border}
            />
            <button style={{ ...S.btn, width:"100%", opacity:sent?0.6:1 }}
              onClick={() => { if(!sent&&msg.trim()){setToast(true);setSent(true);} }}>
              {sent?"✅ Request Sent":"Send Collaboration Request →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════
// SCREEN 4 — UNIVERSITY ONBOARDING + UPLOAD
// ════════════════════════════════════════
function UploadScreen() {
  const [phase, setPhase] = useState("onboard"); // onboard | encode | upload | done
  const [uniName, setUniName] = useState("");
  const [contactEmail, setContactEmail] = useState("");
  const [dept, setDept] = useState("");
  const [files, setFiles] = useState({ text:null, images:null, model:null });
  const [agreed, setAgreed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [toast, setToast] = useState(null);
  const [expandedFaq, setExpandedFaq] = useState(null);
  const [encodeStep, setEncodeStep] = useState(0);

  const fileTypes = [
    { key:"text",   icon:"📄", label:"Text Data (.qcsv.npz)", desc:"Encoded abstracts, grant metadata, thesis summaries", accept:".npz,.zip", color:C.primary },
    { key:"images", icon:"🖼", label:"Image Data (.qimg.npz)", desc:"Encoded research figures, diagrams, imaging data", accept:".npz,.zip", color:C.blueLight },
    { key:"model",  icon:"🧠", label:"Model Weights (.npz)", desc:"Local federated model weights from training round", accept:".npz,.pt,.zip", color:C.amber },
  ];

  const encodeSteps = [
    { icon:"⬇️", title:"Download the encoder", desc:'Click "Get Quantum Encoder" below. It downloads a notebook file (.ipynb) to your computer.' },
    { icon:"🌐", title:"Open in Google Colab", desc:'Go to colab.research.google.com → File → Upload notebook → select the downloaded file. Free, no setup.' },
    { icon:"▶️", title:"Run all cells", desc:'Click Runtime → Run All. Wait ~30 seconds. A clean interface appears — no coding needed.' },
    { icon:"📂", title:"Upload your research files", desc:'Click "Upload Files / ZIP" in the interface. Select your research documents, images, or CSV data files.' },
    { icon:"⚛️", title:"Click Run Encoder", desc:'Hit the green "Run Encoder" button. Your files are quantum-encoded locally in the browser. Nothing leaves your machine.' },
    { icon:"⬇️", title:"Download encoded files", desc:'Click "Download Results". You get a ZIP of .npz files — these are safe to share. Come back here and upload them.' },
  ];

  const faqs = [
    { q:"Do my raw files ever leave my university?", a:"Never. The encoder runs entirely in Google Colab — your files are processed in your browser session. Only the encoded .npz output is downloaded to your machine. Nothing is transmitted to Luminary or any third party during encoding." },
    { q:"What is quantum encoding?", a:"Our encoder uses three quantum computing techniques: Walsh-Hadamard Transform (simulates Hadamard gates on all data qubits simultaneously), quantum phase encoding (converts data values into complex wave functions), and CNOT entanglement (creates quantum-style diffusion between data points). The result is a lossless, encrypted representation of your data." },
    { q:"Can we decode back to the original data?", a:"Yes — the encoding is perfectly lossless. The decoder in our notebook can reconstruct the original files exactly from the .npz output. This proves the integrity of the encoding." },
    { q:"What file types can we encode?", a:"Images: .jpg, .png, .bmp, .tiff, .webp. Structured data: .csv files. For research text (abstracts, grant descriptions), format them as CSV with columns like title, abstract, methodology, domain before encoding." },
    { q:"How long does encoding take?", a:"Typically 10–30 seconds per file depending on size. A batch of 50 research abstracts as CSV takes under a minute. Large imaging datasets may take a few minutes." },
    { q:"What happens after we submit?", a:"Luminary's backend decodes your .npz files, extracts research metadata, and indexes it in our federated network. Within 24–48 hours, researchers globally can discover your institution's work through semantic search and QAOA collaboration matching." },
  ];

  const handleFile = (key, e) => {
    const f = e.target.files[0];
    if (f) setFiles(prev => ({ ...prev, [key]:f }));
  };

  const handleSubmit = async () => {
    if (!uniName || !contactEmail) { setToast({ msg:"Please fill in university name and email", type:"warning" }); return; }
    if (!files.text && !files.images && !files.model) { setToast({ msg:"Please upload at least one encoded data file", type:"warning" }); return; }
    if (!agreed) { setToast({ msg:"Please confirm the data agreement to proceed", type:"warning" }); return; }
    setSubmitting(true);
    await new Promise(r => setTimeout(r, 2800));
    setSubmitting(false);
    setPhase("done");
  };

  // ── PHASE: DONE ──
  if (phase === "done") return (
    <div style={{ maxWidth:600, margin:"80px auto", padding:"0 32px", textAlign:"center", animation:"fadeIn .5s ease" }}>
      <div style={{ width:80, height:80, borderRadius:"50%", background:`${C.success}22`, border:`2px solid ${C.success}`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:32, margin:"0 auto 24px" }}>✓</div>
      <h2 style={{ fontSize:28, fontWeight:800, color:C.white, marginBottom:12 }}>Submission Received</h2>
      <p style={{ fontSize:15, color:C.textSub, lineHeight:1.8, marginBottom:32 }}>
        Thank you, <strong style={{ color:C.amber }}>{uniName}</strong>. Your encoded research data is being processed. Researchers across the Luminary network will be able to discover your institution's work within 24–48 hours.
      </p>
      <div style={{ ...S.card, marginBottom:28, textAlign:"left" }}>
        <div style={{ fontSize:12, fontWeight:700, color:C.amber, marginBottom:14, textTransform:"uppercase", letterSpacing:1 }}>Submission Summary</div>
        {[["University",uniName],["Department",dept||"—"],["Contact",contactEmail],["Text data",files.text?.name||"—"],["Image data",files.images?.name||"—"],["Model weights",files.model?.name||"—"]].map(([k,v]) => (
          <div key={k} style={{ display:"flex", justifyContent:"space-between", padding:"9px 0", borderBottom:`1px solid ${C.border}`, fontSize:13 }}>
            <span style={{ color:C.textMute }}>{k}</span>
            <span style={{ color:C.white }}>{v}</span>
          </div>
        ))}
      </div>
      <div style={{ display:"flex", gap:12, justifyContent:"center" }}>
        <button style={S.btn} onClick={() => { setPhase("onboard"); setUniName(""); setContactEmail(""); setDept(""); setFiles({text:null,images:null,model:null}); setAgreed(false); }}>Submit Another</button>
        <button style={S.btnOutline} onClick={() => setPhase("upload")}>View My Submission</button>
      </div>
    </div>
  );

  // ── PHASE: ONBOARD ──
  if (phase === "onboard") return (
    <div style={{ maxWidth:960, margin:"0 auto", padding:"40px 48px", animation:"fadeIn .4s ease" }}>
      {toast && <Toast msg={toast.msg} type={toast.type} onDone={() => setToast(null)} />}

      <SectionHeader
        tag="University Portal"
        title="Join the Luminary Network"
        sub="Make your institution's research discoverable to PhD students and researchers worldwide — without sharing any raw data."
      />

      {/* Value props */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:16, marginBottom:40 }}>
        {[
          [C.primary, "🔒", "Zero Raw Data Exposure", "Your research content never leaves your servers. Luminary only receives quantum-encoded representations."],
          [C.amber,   "🌐", "Global Discovery", "PhD students at any connected university can find your researchers, datasets, and ongoing work instantly."],
          [C.blueLight,"⚛️","Quantum Protected",  "Our Walsh-Hadamard + phase encoding creates a lossless quantum fingerprint. Decodable only by you."],
        ].map(([col,icon,title,desc]) => (
          <div key={title} style={{ ...S.card, padding:24, borderColor:`${col}33`, borderTopWidth:3 }}>
            <div style={{ fontSize:28, marginBottom:12 }}>{icon}</div>
            <div style={{ fontSize:15, fontWeight:700, color:col, marginBottom:8 }}>{title}</div>
            <div style={{ fontSize:13, color:C.textSub, lineHeight:1.7 }}>{desc}</div>
          </div>
        ))}
      </div>

      {/* Two path CTA */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:20, marginBottom:40 }}>
        <div style={{ ...S.card, padding:28, borderColor:`${C.amber}44`, textAlign:"center" }}>
          <div style={{ fontSize:40, marginBottom:16 }}>⬇️</div>
          <div style={{ fontSize:17, fontWeight:700, color:C.white, marginBottom:8 }}>First time? Get the encoder</div>
          <div style={{ fontSize:13, color:C.textSub, marginBottom:20, lineHeight:1.7 }}>Download our Quantum Encoder notebook. Run it in Google Colab — no coding required. Encode your research files locally and come back to upload.</div>
          <div style={{ display:"flex", gap:10, justifyContent:"center", flexWrap:"wrap" }}>
            <button style={S.btn} onClick={() => setPhase("encode")}>Get Quantum Encoder →</button>
          </div>
        </div>
        <div style={{ ...S.card, padding:28, borderColor:`${C.primary}44`, textAlign:"center" }}>
          <div style={{ fontSize:40, marginBottom:16 }}>📤</div>
          <div style={{ fontSize:17, fontWeight:700, color:C.white, marginBottom:8 }}>Already encoded? Upload now</div>
          <div style={{ fontSize:13, color:C.textSub, marginBottom:20, lineHeight:1.7 }}>Already have your .npz encoded files ready? Skip ahead and submit your data directly to the Luminary network.</div>
          <button style={S.btnOutline} onClick={() => setPhase("upload")}>Upload Encoded Data →</button>
        </div>
      </div>

      {/* Connected universities */}
      <div style={{ ...S.card, marginBottom:32, padding:28 }}>
        <div style={{ fontSize:13, fontWeight:700, color:C.amber, textTransform:"uppercase", letterSpacing:1, marginBottom:16 }}>Currently Connected</div>
        <div style={{ display:"flex", gap:12, flexWrap:"wrap" }}>
          {["Northeastern University","Boston University","MIT","Harvard Medical School","Tufts University"].map(u => (
            <div key={u} style={{ background:`${C.primary}12`, border:`1px solid ${C.primary}22`, borderRadius:8, padding:"8px 16px", fontSize:13, color:C.textSub }}>
              🏛 {u}
            </div>
          ))}
          <div style={{ background:`${C.amber}12`, border:`1px solid ${C.amber}33`, borderRadius:8, padding:"8px 16px", fontSize:13, color:C.amber, fontWeight:600 }}>
            + Your University →
          </div>
        </div>
      </div>

      {/* FAQ */}
      <div>
        <div style={{ fontSize:13, fontWeight:700, color:C.amber, textTransform:"uppercase", letterSpacing:1, marginBottom:16 }}>Frequently Asked Questions</div>
        {faqs.map((f,i) => (
          <div key={i} style={{ ...S.card, marginBottom:10, padding:0, overflow:"hidden", cursor:"pointer" }}
            onClick={() => setExpandedFaq(expandedFaq===i?null:i)}>
            <div style={{ display:"flex", alignItems:"center", justifyContent:"space-between", padding:"16px 20px" }}>
              <span style={{ fontSize:14, color:C.white, fontWeight:500 }}>{f.q}</span>
              <span style={{ color:C.amber, fontSize:18, flexShrink:0, marginLeft:12, transform:expandedFaq===i?"rotate(45deg)":"rotate(0)", transition:"transform .2s" }}>+</span>
            </div>
            {expandedFaq===i && (
              <div style={{ padding:"0 20px 16px", fontSize:13, color:C.textSub, lineHeight:1.8, borderTop:`1px solid ${C.border}` }}>
                <div style={{ paddingTop:14 }}>{f.a}</div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );

  // ── PHASE: ENCODE ──
  if (phase === "encode") return (
    <div style={{ maxWidth:860, margin:"0 auto", padding:"40px 48px", animation:"fadeIn .4s ease" }}>
      <button style={{ ...S.btnOutline, marginBottom:28, fontSize:12 }} onClick={() => setPhase("onboard")}>← Back</button>
      <SectionHeader tag="Step 1 of 2 — Encode" title="Quantum Encoder Setup" sub="Follow these 6 steps. You do not need to write any code. The encoder runs entirely on Google Colab — free and in your browser." />

      {/* Step cards */}
      <div style={{ marginBottom:36 }}>
        {encodeSteps.map((st, i) => (
          <div key={i}
            style={{ display:"flex", gap:20, alignItems:"flex-start", padding:"20px 24px", background: encodeStep===i?`${C.amber}10`:C.bgCard, border:`1px solid ${encodeStep===i?C.amber:C.border}`, borderRadius:12, marginBottom:10, cursor:"pointer", transition:"all .2s" }}
            onClick={() => setEncodeStep(encodeStep===i?-1:i)}
          >
            <div style={{ width:44, height:44, borderRadius:12, background: encodeStep===i?`${C.amber}22`:`${C.primary}15`, border:`1px solid ${encodeStep===i?C.amber:C.primary}33`, display:"flex", alignItems:"center", justifyContent:"center", fontSize:20, flexShrink:0 }}>{st.icon}</div>
            <div style={{ flex:1 }}>
              <div style={{ display:"flex", alignItems:"center", gap:12, marginBottom: encodeStep===i?8:0 }}>
                <span style={{ fontSize:11, color:C.amber, fontWeight:700 }}>Step {i+1}</span>
                <span style={{ fontSize:15, fontWeight:600, color:C.white }}>{st.title}</span>
              </div>
              {encodeStep===i && <div style={{ fontSize:13, color:C.textSub, lineHeight:1.7 }}>{st.desc}</div>}
            </div>
            <span style={{ color:C.textMute, fontSize:14, flexShrink:0 }}>{encodeStep===i?"▲":"▼"}</span>
          </div>
        ))}
      </div>

      {/* Format guide */}
      <div style={{ ...S.card, marginBottom:32, padding:28 }}>
        <div style={{ fontSize:14, fontWeight:700, color:C.white, marginBottom:16 }}>How to Format Your Research Data</div>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:16 }}>
          {[
            { icon:"📄", label:"Text / Abstracts", color:C.primary, items:["Save as .csv file","Columns: title, abstract, methodology, domain, status","One row per research project","UTF-8 encoding"] },
            { icon:"🖼", label:"Images / Figures", color:C.blueLight, items:["Supported: .jpg .png .bmp .tiff .webp","Research diagrams, imaging datasets","Batch upload as .zip folder","No size limit in Colab"] },
            { icon:"🧠", label:"Model Weights", color:C.amber, items:["From your local federated training","Save as .npz or .pt format","Include architecture config if available","Zip multiple files together"] },
          ].map(ft => (
            <div key={ft.label} style={{ background:`${ft.color}08`, border:`1px solid ${ft.color}22`, borderRadius:10, padding:18 }}>
              <div style={{ fontSize:22, marginBottom:8 }}>{ft.icon}</div>
              <div style={{ fontSize:13, fontWeight:700, color:ft.color, marginBottom:10 }}>{ft.label}</div>
              {ft.items.map(item => (
                <div key={item} style={{ fontSize:12, color:C.textSub, padding:"4px 0", borderBottom:`1px solid ${C.border}`, lineHeight:1.5 }}>· {item}</div>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Download + next */}
      <div style={{ display:"flex", alignItems:"center", gap:16, flexWrap:"wrap" }}>
        <button style={S.btn} onClick={() => {
          const a = document.createElement("a");
          a.href = "https://colab.research.google.com/";
          a.target = "_blank";
          a.click();
        }}>⬇ Download Encoder + Open in Colab →</button>
        <button style={S.btnOutline} onClick={() => setPhase("upload")}>I have my encoded files →</button>
        <span style={{ fontSize:12, color:C.textMute }}>Colab is free. No account needed beyond Google.</span>
      </div>
    </div>
  );

  // ── PHASE: UPLOAD ──
  return (
    <div style={{ maxWidth:860, margin:"0 auto", padding:"40px 48px", animation:"fadeIn .4s ease" }}>
      {toast && <Toast msg={toast.msg} type={toast.type} onDone={() => setToast(null)} />}
      <button style={{ ...S.btnOutline, marginBottom:28, fontSize:12 }} onClick={() => setPhase("onboard")}>← Back</button>

      <SectionHeader tag="Step 2 of 2 — Submit" title="Upload Encoded Data" sub="Submit your quantum-encoded .npz files. We accept files from the Luminary Encoder only. Raw data submissions are automatically rejected." />

      {/* Institution details */}
      <div style={{ ...S.card, marginBottom:20, padding:28 }}>
        <div style={{ fontSize:14, fontWeight:700, color:C.white, marginBottom:18 }}>Institution Details</div>
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr", gap:16 }}>
          {[
            { label:"University Name *", val:uniName, set:setUniName, ph:"e.g. Northeastern University" },
            { label:"Department / School", val:dept, set:setDept, ph:"e.g. Khoury College of CS" },
            { label:"Research Admin Email *", val:contactEmail, set:setContactEmail, ph:"research@university.edu" },
          ].map(f => (
            <div key={f.label}>
              <div style={{ fontSize:11, color:C.textMute, marginBottom:6, fontWeight:700, textTransform:"uppercase", letterSpacing:1 }}>{f.label}</div>
              <input value={f.val} onChange={e => f.set(e.target.value)} placeholder={f.ph}
                style={{ ...S.input, width:"100%", display:"block" }}
                onFocus={e => e.target.style.borderColor=C.amber}
                onBlur={e => e.target.style.borderColor=C.border}
              />
            </div>
          ))}
        </div>
      </div>

      {/* File upload zones */}
      <div style={{ ...S.card, marginBottom:20, padding:28 }}>
        <div style={{ fontSize:14, fontWeight:700, color:C.white, marginBottom:6 }}>Upload Encoded Files</div>
        <p style={{ fontSize:13, color:C.textSub, marginBottom:20 }}>Upload the .npz files produced by the Luminary Quantum Encoder. At least one folder required. You can submit all three for maximum discoverability.</p>
        <div style={{ display:"grid", gridTemplateColumns:"repeat(3,1fr)", gap:16 }}>
          {fileTypes.map(ft => (
            <label key={ft.key} htmlFor={`file-${ft.key}`}
              style={{ ...S.card, padding:22, cursor:"pointer", textAlign:"center", borderColor:files[ft.key]?ft.color:C.border, background:files[ft.key]?`${ft.color}08`:C.bgCard, transition:"all .2s", display:"block" }}
              onMouseEnter={e => { e.currentTarget.style.borderColor=ft.color; e.currentTarget.style.background=`${ft.color}08`; }}
              onMouseLeave={e => { e.currentTarget.style.borderColor=files[ft.key]?ft.color:C.border; e.currentTarget.style.background=files[ft.key]?`${ft.color}08`:C.bgCard; }}
            >
              <input id={`file-${ft.key}`} type="file" accept={ft.accept} onChange={e => handleFile(ft.key,e)} style={{ display:"none" }} />
              <div style={{ fontSize:32, marginBottom:10 }}>{ft.icon.split(" ")[0]}</div>
              <div style={{ fontSize:13, fontWeight:700, color:C.white, marginBottom:6 }}>{ft.label}</div>
              <div style={{ fontSize:12, color:C.textSub, marginBottom:14, lineHeight:1.5 }}>{ft.desc}</div>
              {files[ft.key]
                ? <div style={{ fontSize:12, color:ft.color, fontWeight:600, background:`${ft.color}18`, padding:"5px 12px", borderRadius:20 }}>✓ {files[ft.key].name}</div>
                : <div style={{ fontSize:12, color:C.textMute, border:`1px dashed ${C.border}`, padding:"6px 12px", borderRadius:8 }}>Click to upload</div>
              }
            </label>
          ))}
        </div>
      </div>

      {/* Agreement + submit */}
      <div style={{ ...S.card, padding:24 }}>
        <Toggle value={agreed} onChange={setAgreed} label="I confirm all uploaded files are quantum-encoded using the Luminary Encoder. No raw personal, student, or research data is included. I am authorized to submit on behalf of my institution." />
        <div style={{ marginTop:20, display:"flex", alignItems:"center", gap:16, flexWrap:"wrap" }}>
          <button style={{ ...S.btn, opacity:submitting?0.7:1, minWidth:220 }} onClick={handleSubmit} disabled={submitting}>
            {submitting ? "⏳ Processing..." : "Submit to Luminary Network →"}
          </button>
          <span style={{ fontSize:12, color:C.textMute }}>Indexed within 24–48 hours · FERPA compliant · IRB safe</span>
        </div>
      </div>
    </div>
  );
}

// ════════════════════════════════════════
// APP ROOT
// ════════════════════════════════════════
export default function App() {
  const [screen, setScreen]     = useState("search");
  const [query, setQuery]       = useState("");
  const [selected, setSelected] = useState(null);

  const goSearch  = ()  => setScreen("search");
  const goResults = (q) => { setQuery(q); setScreen("results"); };
  const goProfile = (r) => { setSelected(r); setScreen("profile"); };
  const goNav     = (s) => setScreen(s);

  return (
    <div style={S.app}>
      <CosmicBg />
      <div style={{ position:"relative", zIndex:1 }}>
      <style>{`
        * { box-sizing:border-box; margin:0; padding:0; }
        html,body { width:100%; background:${C.bg}; }
        ::-webkit-scrollbar { width:5px; }
        ::-webkit-scrollbar-track { background:${C.bg}; }
        ::-webkit-scrollbar-thumb { background:rgba(255,69,0,0.4); border-radius:3px; }
        textarea::placeholder,input::placeholder { color:${C.textMute}; }
        @keyframes slideUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
        @keyframes fadeIn  { from{opacity:0} to{opacity:1} }
        @keyframes pulse   { 0%,100%{opacity:0.7} 50%{opacity:1} }
        a { color:${C.amber}; }
      `}</style>

      {/* NAV */}
      <nav style={S.nav}>
        <div style={S.logo}>
          <div style={S.logoMark}>L</div>
          <span style={{ color:C.white, letterSpacing:"-0.5px" }}>Luminary</span>
          <span style={{ fontSize:11, color:C.textMute, fontWeight:400, marginLeft:4 }}>by Quaffle</span>
        </div>

        <div style={{ display:"flex", gap:4 }}>
          {NAV_ITEMS.map(item => {
            const s = item.toLowerCase();
            const active = screen===s || (s==="results"&&screen==="profile");
            return (
              <span key={item} onClick={() => {
                if(s==="results"&&query) setScreen("results");
                else if(s!=="results") setScreen(s);
              }} style={active ? S.navActive : S.navLink}>{item}</span>
            );
          })}
        </div>

        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <div style={{ textAlign:"right" }}>
            <div style={{ fontSize:12, color:C.textSub, fontWeight:600 }}>Team Quaffle</div>
            <div style={{ fontSize:10, color:C.textMute }}>Protothon 8.0</div>
          </div>
          <div style={S.avatar}>Q</div>
        </div>
      </nav>

      {screen==="search"  && <SearchScreen onSearch={goResults} onNav={goNav} />}
      {screen==="results" && <ResultsScreen query={query} onBack={goSearch} onSelect={goProfile} />}
      {screen==="profile" && selected && <ProfileScreen researcher={selected} onBack={() => setScreen("results")} />}
      {screen==="upload"  && <UploadScreen />}
      {screen==="network" && <NetworkPlaceholder />}
      </div>
    </div>
  );
}

function NetworkPlaceholder() {
  return (
    <div style={{ padding:"40px 48px", maxWidth:900, margin:"0 auto" }}>
      <SectionHeader tag="Live Network" title="Federated Learning Network" sub="Each university trains locally. Only encrypted model weights are shared. Raw data never leaves any node." />
      <div style={{ background:"#0d1f35", border:"1px solid #1a3a5a", borderRadius:14, padding:40, textAlign:"center", color:"#7aacc8" }}>
        Import FederatedScreen component here — already built in FederatedScreen.jsx
      </div>
    </div>
  );
}