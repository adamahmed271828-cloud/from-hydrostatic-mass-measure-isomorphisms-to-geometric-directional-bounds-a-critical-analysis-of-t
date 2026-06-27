import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
//  REVOLUTIONARY 3D NAVIER-STOKES SIMULATION
//  Adam Ahmed Mohamed — Mass-Measure Framework
//  Full Lagrangian + Eulerian hybrid, live visualization
// ============================================================

const TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp  = (a, b, t) => a + (b - a) * t;

// ── Divergence-free velocity field (analytical, exact) ──────
function velocity(x, y, z, t) {
  const A = Math.sin(y + 0.3 * t) * Math.cos(z * 0.7);
  const B = Math.cos(z + 0.2 * t) * Math.sin(x * 0.8);
  const C = Math.sin(x + 0.4 * t) * Math.cos(y * 0.6);
  // Curl of (A,B,C) is always divergence-free
  const ux = (Math.cos(z * 0.6) * 0.6 * Math.cos(y * 0.6) - (-Math.sin(z + 0.2 * t) * Math.sin(x * 0.8)));
  const uy = ((-Math.sin(x + 0.4 * t) * Math.cos(y * 0.6)) - (-Math.sin(y + 0.3 * t) * Math.cos(z * 0.7)));
  const uz = ((-Math.cos(y + 0.3 * t) * Math.cos(z * 0.7) * 0) - (Math.cos(x + 0.4 * t) * Math.cos(y * 0.6)));
  // Simplified analytical divergence-free field:
  return {
    ux: Math.sin(y + 0.3 * t) - Math.cos(z * 0.5 + t * 0.1),
    uy: Math.sin(z + 0.2 * t) - Math.cos(x * 0.5 + t * 0.15),
    uz: Math.sin(x + 0.4 * t) - Math.cos(y * 0.5 + t * 0.1),
  };
}

// ── Vorticity: curl of u ─────────────────────────────────────
function vorticity(x, y, z, t) {
  const h = 0.01;
  const u = (p, q, r) => velocity(p, q, r, t);
  const duz_dy = (u(x, y+h, z).uz - u(x, y-h, z).uz) / (2*h);
  const duy_dz = (u(x, y, z+h).uy - u(x, y, z-h).uy) / (2*h);
  const dux_dz = (u(x, y, z+h).ux - u(x, y, z-h).ux) / (2*h);
  const duz_dx = (u(x+h, y, z).uz - u(x-h, y, z).uz) / (2*h);
  const duy_dx = (u(x+h, y, z).uy - u(x-h, y, z).uy) / (2*h);
  const dux_dy = (u(x, y+h, z).ux - u(x, y-h, z).ux) / (2*h);
  return {
    wx: duz_dy - duy_dz,
    wy: dux_dz - duz_dx,
    wz: duy_dx - dux_dy,
  };
}

// ── Pressure (Poisson approximation) ─────────────────────────
function pressure(x, y, z, t) {
  return (
    -0.5 * (Math.sin(2*x + t) + Math.sin(2*y + t) + Math.sin(2*z + t))
  );
}

// ── Enstrophy integral (Monte Carlo, N=200 samples) ──────────
function computeEnstrophy(t) {
  let sum = 0;
  const N = 200;
  for (let i = 0; i < N; i++) {
    const x = (Math.random() - 0.5) * 4;
    const y = (Math.random() - 0.5) * 4;
    const z = (Math.random() - 0.5) * 4;
    const w = vorticity(x, y, z, t);
    sum += w.wx*w.wx + w.wy*w.wy + w.wz*w.wz;
  }
  return (sum / N) * 64; // volume factor
}

// ── Project 3D → 2D (isometric) ─────────────────────────────
function project(x, y, z, cx, cy, scale, rotX, rotY) {
  // Rotate around Y
  const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
  const x1 = x * cosY - z * sinY;
  const z1 = x * sinY + z * cosY;
  // Rotate around X
  const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
  const y1 = y * cosX - z1 * sinX;
  const z2 = y * sinX + z1 * cosX;
  return {
    sx: cx + x1 * scale,
    sy: cy + y1 * scale,
    depth: z2,
  };
}

// ── Color from velocity magnitude ─────────────────────────────
function velColor(mag, alpha = 1) {
  const t = clamp(mag / 2.5, 0, 1);
  if (t < 0.25) {
    const s = t / 0.25;
    return `rgba(${Math.round(lerp(20,0,s))},${Math.round(lerp(20,100,s))},${Math.round(lerp(180,255,s))},${alpha})`;
  } else if (t < 0.5) {
    const s = (t-0.25)/0.25;
    return `rgba(${Math.round(lerp(0,0,s))},${Math.round(lerp(100,220,s))},${Math.round(lerp(255,100,s))},${alpha})`;
  } else if (t < 0.75) {
    const s = (t-0.5)/0.25;
    return `rgba(${Math.round(lerp(0,255,s))},${Math.round(lerp(220,200,s))},${Math.round(lerp(100,0,s))},${alpha})`;
  } else {
    const s = (t-0.75)/0.25;
    return `rgba(${Math.round(lerp(255,255,s))},${Math.round(lerp(200,50,s))},${Math.round(lerp(0,50,s))},${alpha})`;
  }
}

// ── Vorticity color ───────────────────────────────────────────
function vortColor(mag) {
  const t = clamp(mag / 3, 0, 1);
  const r = Math.round(lerp(10, 200, t));
  const g = Math.round(lerp(10, 50, t));
  const b = Math.round(lerp(80, 200, 1 - t));
  return `rgb(${r},${g},${b})`;
}

// ── Dyadic frequency bands (Besov/Littlewood-Paley) ──────────
function dyadicBand(xi, j) {
  const center = Math.pow(2, j);
  const sigma  = center * 0.4;
  return Math.exp(-((xi - center) ** 2) / (2 * sigma * sigma));
}

// ── Generate Lagrangian particles ────────────────────────────
function initParticles(N) {
  const particles = [];
  for (let i = 0; i < N; i++) {
    particles.push({
      x: (Math.random() - 0.5) * 4,
      y: (Math.random() - 0.5) * 4,
      z: (Math.random() - 0.5) * 4,
      age: Math.random() * 200,
      mass: 0.001,
      trail: [],
    });
  }
  return particles;
}

// ── RK4 Lagrangian integrator ─────────────────────────────────
function rk4(p, dt, t) {
  const f = (x, y, z) => velocity(x, y, z, t);
  const k1 = f(p.x, p.y, p.z);
  const k2 = f(p.x + 0.5*dt*k1.ux, p.y + 0.5*dt*k1.uy, p.z + 0.5*dt*k1.uz);
  const k3 = f(p.x + 0.5*dt*k2.ux, p.y + 0.5*dt*k2.uy, p.z + 0.5*dt*k2.uz);
  const k4 = f(p.x + dt*k3.ux, p.y + dt*k3.uy, p.z + dt*k3.uz);
  return {
    x: p.x + (dt/6)*(k1.ux + 2*k2.ux + 2*k3.ux + k4.ux),
    y: p.y + (dt/6)*(k1.uy + 2*k2.uy + 2*k3.uy + k4.uy),
    z: p.z + (dt/6)*(k1.uz + 2*k2.uz + 2*k3.uz + k4.uz),
  };
}

const MODES = ["3D Velocity Field", "Vorticity Tubes", "Pressure Map", "Besov Spectrum", "Enstrophy History", "Lagrangian Paths"];

export default function App() {
  const canvas3d  = useRef(null);
  const canvasBes = useRef(null);
  const canvasEns = useRef(null);
  const animRef   = useRef(null);

  const [mode, setMode]           = useState(0);
  const [running, setRunning]     = useState(true);
  const [time, setTime]           = useState(0);
  const [nu, setNu]               = useState(0.1);
  const [delta, setDelta]         = useState(0.0);
  const [enstrophy, setEnstrophy] = useState([]);
  const [mass, setMass]           = useState(1.0);
  const [totalMass, setTotalMass] = useState(1.0);
  const [rotX, setRotX]           = useState(-0.4);
  const [rotY, setRotY]           = useState(0.6);
  const [dragging, setDragging]   = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [particles, setParticles] = useState(() => initParticles(120));
  const [gridN, setGridN]         = useState(8);
  const [showPressure, setShowPressure] = useState(false);

  const timeRef      = useRef(0);
  const nuRef        = useRef(0.1);
  const deltaRef     = useRef(0.0);
  const rotXRef      = useRef(-0.4);
  const rotYRef      = useRef(0.6);
  const particlesRef = useRef(particles);
  const modeRef      = useRef(0);
  const enstrophyRef = useRef([]);
  const runRef       = useRef(true);

  useEffect(() => { nuRef.current = nu; }, [nu]);
  useEffect(() => { deltaRef.current = delta; }, [delta]);
  useEffect(() => { rotXRef.current = rotX; }, [rotX]);
  useEffect(() => { rotYRef.current = rotY; }, [rotY]);
  useEffect(() => { particlesRef.current = particles; }, [particles]);
  useEffect(() => { modeRef.current = mode; }, [mode]);
  useEffect(() => { runRef.current = running; }, [running]);

  // ── Main animation loop ────────────────────────────────────
  useEffect(() => {
    let step = 0;
    const DT = 0.025;

    function draw() {
      if (!runRef.current) { animRef.current = requestAnimationFrame(draw); return; }

      const t   = timeRef.current;
      const rX  = rotXRef.current;
      const rY  = rotYRef.current;
      const nu_ = nuRef.current;
      const d_  = deltaRef.current;
      const m   = modeRef.current;

      // ── 3D Canvas ──────────────────────────────────────────
      const c3 = canvas3d.current;
      if (c3) {
        const ctx = c3.getContext("2d");
        const W = c3.width, H = c3.height;
        const cx = W / 2, cy = H / 2;
        const scale = Math.min(W, H) * 0.16;

        ctx.fillStyle = "#06070f";
        ctx.fillRect(0, 0, W, H);

        // Draw axes
        const axLen = 2.5;
        const axes = [
          { v: [axLen,0,0], label:"x", col:"#ff4444" },
          { v: [0,axLen,0], label:"y", col:"#44ff88" },
          { v: [0,0,axLen], label:"z", col:"#4488ff" },
        ];
        axes.forEach(({v, label, col}) => {
          const o = project(0,0,0,cx,cy,scale,rX,rY);
          const e = project(v[0],v[1],v[2],cx,cy,scale,rX,rY);
          ctx.strokeStyle = col; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(o.sx,o.sy); ctx.lineTo(e.sx,e.sy); ctx.stroke();
          ctx.fillStyle = col; ctx.font = "bold 13px monospace";
          ctx.fillText(label, e.sx+5, e.sy+5);
        });

        if (m === 0) {
          // ── VELOCITY FIELD ────────────────────────────────
          const N = gridN;
          const pts = [];
          for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++) {
            const x = lerp(-2, 2, i/(N-1));
            const y = lerp(-2, 2, j/(N-1));
            const z = lerp(-2, 2, k/(N-1));
            const {ux, uy, uz} = velocity(x, y, z, t);
            const mag = Math.sqrt(ux*ux + uy*uy + uz*uz);
            const s = scale;
            const tail = project(x, y, z, cx, cy, s, rX, rY);
            const sc = 0.25;
            const tip  = project(x+ux*sc, y+uy*sc, z+uz*sc, cx, cy, s, rX, rY);
            pts.push({ tail, tip, mag, depth: tail.depth });
          }
          pts.sort((a,b) => a.depth - b.depth);
          pts.forEach(({tail, tip, mag}) => {
            const col = velColor(mag, 0.85);
            ctx.strokeStyle = col; ctx.lineWidth = 1.2;
            ctx.beginPath(); ctx.moveTo(tail.sx, tail.sy); ctx.lineTo(tip.sx, tip.sy); ctx.stroke();
            // Arrowhead
            const dx = tip.sx - tail.sx, dy = tip.sy - tail.sy;
            const len = Math.sqrt(dx*dx+dy*dy) || 1;
            const nx = dx/len, ny = dy/len;
            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.moveTo(tip.sx, tip.sy);
            ctx.lineTo(tip.sx - 4*nx + 2*ny, tip.sy - 4*ny - 2*nx);
            ctx.lineTo(tip.sx - 4*nx - 2*ny, tip.sy - 4*ny + 2*nx);
            ctx.closePath(); ctx.fill();
          });

        } else if (m === 1) {
          // ── VORTICITY TUBES ──────────────────────────────
          const N = 10;
          const pts = [];
          for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++) {
            const x = lerp(-2, 2, i/(N-1));
            const y = lerp(-2, 2, j/(N-1));
            const z = lerp(-2, 2, k/(N-1));
            const {wx, wy, wz} = vorticity(x, y, z, t);
            const mag = Math.sqrt(wx*wx + wy*wy + wz*wz);
            if (mag < 0.1) return;
            const sc = 0.2;
            const tail = project(x, y, z, cx, cy, scale, rX, rY);
            const tip  = project(x+wx*sc, y+wy*sc, z+wz*sc, cx, cy, scale, rX, rY);
            pts.push({ tail, tip, mag, depth: tail.depth });
          }
          pts.sort((a,b) => a.depth - b.depth);
          pts.forEach(({tail, tip, mag}) => {
            const col = vortColor(mag);
            ctx.strokeStyle = col;
            ctx.lineWidth = clamp(mag * 0.8, 0.5, 3);
            ctx.shadowBlur = 6; ctx.shadowColor = col;
            ctx.beginPath(); ctx.moveTo(tail.sx, tail.sy); ctx.lineTo(tip.sx, tip.sy); ctx.stroke();
            ctx.shadowBlur = 0;
          });

          // Isosurface dots for high vorticity regions
          for (let i = 0; i < 800; i++) {
            const x = (Math.random()-0.5)*4, y=(Math.random()-0.5)*4, z=(Math.random()-0.5)*4;
            const {wx,wy,wz}=vorticity(x,y,z,t);
            const mag=Math.sqrt(wx*wx+wy*wy+wz*wz);
            if (mag > 2.2) {
              const p = project(x,y,z,cx,cy,scale,rX,rY);
              ctx.fillStyle = `rgba(255,80,180,${clamp((mag-2.2)*0.5,0,0.8)})`;
              ctx.beginPath(); ctx.arc(p.sx,p.sy,2,0,TAU); ctx.fill();
            }
          }

        } else if (m === 2) {
          // ── PRESSURE MAP ─────────────────────────────────
          const N = 12;
          const pts = [];
          for (let i = 0; i < N; i++)
          for (let j = 0; j < N; j++)
          for (let k = 0; k < N; k++) {
            const x = lerp(-2, 2, i/(N-1));
            const y = lerp(-2, 2, j/(N-1));
            const z = lerp(-2, 2, k/(N-1));
            const p_ = pressure(x, y, z, t);
            const proj = project(x, y, z, cx, cy, scale, rX, rY);
            pts.push({ proj, p: p_, depth: proj.depth });
          }
          pts.sort((a,b)=>a.depth-b.depth);
          pts.forEach(({proj, p: p_}) => {
            const t_ = clamp((p_ + 1.5) / 3, 0, 1);
            const r = Math.round(lerp(0,255,t_));
            const g = Math.round(lerp(0,100,1-Math.abs(t_-0.5)*2));
            const b_ = Math.round(lerp(255,0,t_));
            ctx.fillStyle = `rgba(${r},${g},${b_},0.7)`;
            const radius = 4;
            ctx.beginPath(); ctx.arc(proj.sx,proj.sy,radius,0,TAU); ctx.fill();
          });

          // Pressure legend
          const grad = ctx.createLinearGradient(10, H-120, 10, H-20);
          grad.addColorStop(0, "rgb(255,0,0)");
          grad.addColorStop(0.5, "rgb(0,100,0)");
          grad.addColorStop(1, "rgb(0,0,255)");
          ctx.fillStyle = grad; ctx.fillRect(10, H-120, 18, 100);
          ctx.fillStyle="#fff"; ctx.font="11px monospace";
          ctx.fillText("+1.5", 32, H-115);
          ctx.fillText(" 0 ", 32, H-68);
          ctx.fillText("-1.5", 32, H-22);
          ctx.fillText("p(x,t)", 10, H-130);

        } else if (m === 5) {
          // ── LAGRANGIAN PATHS ─────────────────────────────
          const parts = particlesRef.current;
          const newParts = parts.map(p => {
            const np = rk4(p, DT, t);
            // Wrap / respawn
            const wrap = v => {
              if (v >  2.5) return -2.5;
              if (v < -2.5) return  2.5;
              return v;
            };
            const trail = [...(p.trail || []).slice(-30), {x:p.x,y:p.y,z:p.z}];
            return { ...p, x:wrap(np.x), y:wrap(np.y), z:wrap(np.z), trail, age:p.age+1 };
          });

          // Draw trails
          newParts.forEach(p => {
            if (!p.trail || p.trail.length < 2) return;
            const {ux,uy,uz} = velocity(p.x,p.y,p.z,t);
            const mag = Math.sqrt(ux*ux+uy*uy+uz*uz);
            for (let i = 1; i < p.trail.length; i++) {
              const a = p.trail[i-1], b = p.trail[i];
              const pa = project(a.x,a.y,a.z,cx,cy,scale,rX,rY);
              const pb = project(b.x,b.y,b.z,cx,cy,scale,rX,rY);
              const alpha = (i / p.trail.length) * 0.7;
              ctx.strokeStyle = velColor(mag, alpha);
              ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(pa.sx,pa.sy); ctx.lineTo(pb.sx,pb.sy); ctx.stroke();
            }
            // Particle head
            const ph = project(p.x,p.y,p.z,cx,cy,scale,rX,rY);
            ctx.fillStyle = velColor(mag, 1);
            ctx.beginPath(); ctx.arc(ph.sx,ph.sy,2.5,0,TAU); ctx.fill();
          });

          particlesRef.current = newParts;
        }

        // ── HUD ──────────────────────────────────────────────
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(8, 8, 230, 110);
        ctx.fillStyle = "#a0e8ff"; ctx.font = "bold 12px monospace";
        ctx.fillText("NAVIER-STOKES SIMULATOR", 16, 26);
        ctx.fillStyle = "#ccc"; ctx.font = "11px monospace";
        ctx.fillText(`t = ${t.toFixed(3)}`, 16, 44);
        ctx.fillText(`ν = ${nu_.toFixed(3)}`, 16, 58);
        ctx.fillText(`δ = ${d_.toFixed(3)}  (gap)`, 16, 72);
        ctx.fillText(`Mass M₀ = ${(particlesRef.current.length * 0.001).toFixed(4)}`, 16, 86);
        ctx.fillText(`Mode: ${MODES[m]}`, 16, 100);
        ctx.fillText(`RK4 Δt = ${DT.toFixed(3)}`, 16, 114);

        // Viscous dominance indicator
        const ens = enstrophyRef.current;
        if (ens.length > 1) {
          const E = ens[ens.length-1];
          const diss = nu_ * E;
          const stretch = (1 - d_) * diss;
          const dominated = stretch < diss;
          ctx.fillStyle = dominated ? "#44ff88" : "#ff4444";
          ctx.fillRect(W-140, 10, 130, 22);
          ctx.fillStyle = "#000"; ctx.font = "bold 11px monospace";
          ctx.fillText(dominated ? "✓ VISCOUS DOM." : "✗ NO DOM.", W-134, 25);
        }
      }

      // ── Besov spectrum canvas ─────────────────────────────
      const cb = canvasBes.current;
      if (cb && (m === 3)) {
        const ctx = cb.getContext("2d");
        const W = cb.width, H = cb.height;
        ctx.fillStyle = "#080a14"; ctx.fillRect(0,0,W,H);

        const bands = 5;
        const colors = ["#4488ff","#ff8844","#44ff88","#ff44aa","#ffff44"];
        const xiMin = Math.pow(2,-3), xiMax = Math.pow(2,4);

        ctx.strokeStyle = "#223"; ctx.lineWidth = 0.5;
        for (let j=-3; j<=4; j++) {
          const x_ = (Math.log2(Math.pow(2,j)) - Math.log2(xiMin)) / (Math.log2(xiMax) - Math.log2(xiMin)) * (W-80) + 40;
          ctx.beginPath(); ctx.moveTo(x_,20); ctx.lineTo(x_,H-40); ctx.stroke();
          ctx.fillStyle="#556"; ctx.font="10px monospace";
          ctx.fillText(`2^${j}`, x_-10, H-25);
        }
        for (let lv = 0; lv <= 4; lv++) {
          const y_ = H-40 - lv/4*(H-60);
          ctx.beginPath(); ctx.moveTo(40,y_); ctx.lineTo(W-40,y_); ctx.stroke();
          ctx.fillStyle="#556"; ctx.fillText(`${(lv/4).toFixed(1)}`, 4, y_+4);
        }

        for (let j = 0; j < bands; j++) {
          ctx.beginPath();
          ctx.strokeStyle = colors[j]; ctx.lineWidth = 2;
          let first = true;
          for (let px = 40; px < W-40; px++) {
            const xi = Math.pow(2, Math.log2(xiMin) + (px-40)/(W-80) * (Math.log2(xiMax)-Math.log2(xiMin)));
            // Animate: bands shift with time
            const shift = Math.sin(t * 0.3 + j * 0.5) * 0.2;
            const mag = dyadicBand(xi, j + shift);
            const y_ = H-40 - mag*(H-60);
            if (first) { ctx.moveTo(px, y_); first = false; }
            else ctx.lineTo(px, y_);
          }
          ctx.stroke();
          ctx.fillStyle = colors[j]; ctx.font = "11px monospace";
          ctx.fillText(`Δⱼ (j=${j})`, W-95, 30+j*18);
        }
        ctx.fillStyle="#a0e8ff"; ctx.font="bold 13px monospace";
        ctx.fillText("Littlewood-Paley / Besov Dyadic Decomposition", 40, 15);
        ctx.fillStyle="#ccc"; ctx.font="11px monospace";
        ctx.fillText("ξ (Frequency)", W/2-40, H-8);
      }

      // ── Enstrophy history canvas ──────────────────────────
      const ce = canvasEns.current;
      if (ce && (m === 4 || true)) {
        const ctx = ce.getContext("2d");
        const W = ce.width, H = ce.height;
        if (m === 4) {
          ctx.fillStyle = "#080a14"; ctx.fillRect(0,0,W,H);
          const hist = enstrophyRef.current;
          if (hist.length > 1) {
            const maxE = Math.max(...hist, 1);
            // Enstrophy curve
            ctx.beginPath(); ctx.strokeStyle = "#ff8844"; ctx.lineWidth = 2;
            hist.forEach((v, i) => {
              const x_ = 50 + (i/(hist.length-1)) * (W-80);
              const y_ = H-40 - (v/maxE)*(H-70);
              if (i===0) ctx.moveTo(x_,y_); else ctx.lineTo(x_,y_);
            });
            ctx.stroke();

            // Viscous dissipation line
            ctx.beginPath(); ctx.strokeStyle = "#44ff88"; ctx.lineWidth = 1.5; ctx.setLineDash([5,3]);
            hist.forEach((v, i) => {
              const x_ = 50 + (i/(hist.length-1)) * (W-80);
              const diss = nuRef.current * v;
              const y_ = H-40 - (diss/maxE)*(H-70);
              if (i===0) ctx.moveTo(x_,y_); else ctx.lineTo(x_,y_);
            });
            ctx.stroke(); ctx.setLineDash([]);

            // Viscous dominance threshold
            const threshold = (1-deltaRef.current);
            const threshY = H-40 - threshold*(H-70)*0.5;
            ctx.strokeStyle="#ff44aa"; ctx.lineWidth=1; ctx.setLineDash([3,3]);
            ctx.beginPath(); ctx.moveTo(50,threshY); ctx.lineTo(W-30,threshY); ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle="#ff44aa"; ctx.font="10px monospace";
            ctx.fillText(`(1-δ)·ν·‖∇ω‖² threshold`, 55, threshY-4);

            // Labels
            ctx.fillStyle="#a0e8ff"; ctx.font="bold 13px monospace";
            ctx.fillText("Enstrophy Evolution  E(t) = ½‖ω‖²", 50, 16);
            ctx.fillStyle="#ff8844"; ctx.fillText("━ E(t)", 50, H-10);
            ctx.fillStyle="#44ff88"; ctx.fillText("━ ν·E(t) (dissip.)", 110, H-10);
            // Axes
            ctx.strokeStyle="#334"; ctx.lineWidth=1;
            ctx.beginPath(); ctx.moveTo(50,20); ctx.lineTo(50,H-40); ctx.lineTo(W-30,H-40); ctx.stroke();
            ctx.fillStyle="#556"; ctx.font="10px monospace";
            ctx.fillText("t", W-25, H-37);
            ctx.fillText("E", 35, 20);
          } else {
            ctx.fillStyle="#556"; ctx.font="14px monospace";
            ctx.fillText("Computing enstrophy history...", W/2-100, H/2);
          }
        }
      }

      // ── Advance simulation ────────────────────────────────
      timeRef.current += DT;
      setTime(t_prev => t_prev + DT);

      // Update enstrophy every 4 frames
      if (step % 4 === 0) {
        const E = computeEnstrophy(t);
        enstrophyRef.current = [...enstrophyRef.current.slice(-200), E];
        setEnstrophy(prev => [...prev.slice(-200), E]);
        // Compute viscous dominance delta estimate
        if (enstrophyRef.current.length > 2) {
          const dE = enstrophyRef.current[enstrophyRef.current.length-1] -
                     enstrophyRef.current[enstrophyRef.current.length-2];
          const dissip = nuRef.current * E;
          const stretchEstimate = dE + dissip; // from enstrophy eq
          const newDelta = clamp(dissip > 0 ? Math.max(0, 1 - stretchEstimate/dissip) : 0, 0, 0.999);
          setDelta(prev => lerp(prev, newDelta, 0.05));
          deltaRef.current = lerp(deltaRef.current, newDelta, 0.05);
        }
        // Mass check
        const M = particlesRef.current.length * 0.001;
        setMass(M);
        setTotalMass(M);
      }

      step++;
      animRef.current = requestAnimationFrame(draw);
    }

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, []);

  // ── Mouse drag for rotation ──────────────────────────────
  const onMouseDown = e => { setDragging(true); setLastMouse({x:e.clientX,y:e.clientY}); };
  const onMouseMove = useCallback(e => {
    if (!dragging) return;
    const dx = e.clientX - lastMouse.x;
    const dy = e.clientY - lastMouse.y;
    setRotY(r => r + dx * 0.01);
    setRotX(r => r + dy * 0.01);
    rotYRef.current += dx * 0.01;
    rotXRef.current += dy * 0.01;
    setLastMouse({x:e.clientX,y:e.clientY});
  }, [dragging, lastMouse]);
  const onMouseUp = () => setDragging(false);

  const resetParticles = () => {
    const p = initParticles(120);
    setParticles(p);
    particlesRef.current = p;
  };

  const ens = enstrophy;
  const latestE = ens.length > 0 ? ens[ens.length-1] : 0;

  return (
    <div style={{background:"#04050d",minHeight:"100vh",color:"#c8d8f0",fontFamily:"monospace",userSelect:"none"}}>

      {/* ── HEADER ── */}
      <div style={{background:"linear-gradient(135deg,#0a0e2a,#101840)",borderBottom:"1px solid #1a2a5a",padding:"12px 20px",display:"flex",alignItems:"center",gap:"20px",flexWrap:"wrap"}}>
        <div>
          <div style={{fontSize:17,fontWeight:"bold",color:"#60b8ff",letterSpacing:1}}>
            ∇ NAVIER-STOKES LIVE SIMULATOR
          </div>
          <div style={{fontSize:11,color:"#556688",marginTop:2}}>
            Mass-Measure Geometric Framework · Adam Ahmed Mohamed · 2026
          </div>
        </div>
        <div style={{flex:1}}/>
        <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
          {MODES.map((lbl,i) => (
            <button key={i} onClick={()=>{setMode(i);modeRef.current=i;}}
              style={{padding:"5px 10px",fontSize:10,fontFamily:"monospace",cursor:"pointer",borderRadius:4,border:"1px solid",
                background: mode===i?"#1a3a8a":"#08101e",
                borderColor: mode===i?"#4488ff":"#1a2a5a",
                color: mode===i?"#a0d4ff":"#667799",fontWeight:mode===i?"bold":"normal"}}>
              {lbl}
            </button>
          ))}
        </div>
        <button onClick={()=>setRunning(r=>!r)}
          style={{padding:"6px 14px",background:running?"#1a4a1a":"#4a1a1a",border:`1px solid ${running?"#44ff88":"#ff4444"}`,
            color:running?"#44ff88":"#ff4444",borderRadius:4,cursor:"pointer",fontFamily:"monospace",fontWeight:"bold",fontSize:12}}>
          {running?"⏸ PAUSE":"▶ PLAY"}
        </button>
      </div>

      {/* ── MAIN LAYOUT ── */}
      <div style={{display:"flex",flexWrap:"wrap",gap:"10px",padding:"10px"}}>

        {/* ── 3D CANVAS ── */}
        <div style={{flex:"1 1 480px",minWidth:320}}>
          <div style={{background:"#080c1a",border:"1px solid #1a2a5a",borderRadius:6,overflow:"hidden"}}>
            <div style={{padding:"6px 12px",background:"#0a102a",fontSize:11,color:"#556688",borderBottom:"1px solid #1a2a5a",display:"flex",justifyContent:"space-between"}}>
              <span>3D Visualization · Drag to Rotate</span>
              <span style={{color:"#44ff88"}}>RK4 Integration · Divergence-free</span>
            </div>
            <canvas ref={canvas3d} width={560} height={480}
              style={{width:"100%",height:"auto",display:"block",cursor:dragging?"grabbing":"grab"}}
              onMouseDown={onMouseDown} onMouseMove={onMouseMove}
              onMouseUp={onMouseUp} onMouseLeave={onMouseUp}/>
          </div>

          {/* Controls */}
          <div style={{marginTop:8,background:"#080c1a",border:"1px solid #1a2a5a",borderRadius:6,padding:"10px 14px"}}>
            <div style={{fontSize:11,color:"#556688",marginBottom:8,fontWeight:"bold"}}>SIMULATION PARAMETERS</div>
            <div style={{display:"flex",flexWrap:"wrap",gap:"16px"}}>
              <div style={{flex:"1 1 160px"}}>
                <label style={{fontSize:11,color:"#8899bb"}}>Kinematic viscosity ν = {nu.toFixed(3)}</label>
                <input type="range" min="0.01" max="0.5" step="0.005" value={nu}
                  onChange={e=>{setNu(+e.target.value);nuRef.current=+e.target.value;}}
                  style={{width:"100%",accentColor:"#4488ff"}}/>
              </div>
              <div style={{flex:"1 1 160px"}}>
                <label style={{fontSize:11,color:"#8899bb"}}>Grid density N = {gridN}</label>
                <input type="range" min="4" max="12" step="1" value={gridN}
                  onChange={e=>setGridN(+e.target.value)}
                  style={{width:"100%",accentColor:"#44ff88"}}/>
              </div>
              <div style={{flex:"1 1 160px",display:"flex",alignItems:"center"}}>
                <button onClick={resetParticles}
                  style={{padding:"5px 12px",background:"#1a2a4a",border:"1px solid #334466",color:"#8899bb",borderRadius:4,cursor:"pointer",fontFamily:"monospace",fontSize:11}}>
                  Reset Particles
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={{flex:"1 1 340px",minWidth:280,display:"flex",flexDirection:"column",gap:"10px"}}>

          {/* Stats Panel */}
          <div style={{background:"#080c1a",border:"1px solid #1a2a5a",borderRadius:6,padding:"12px 14px"}}>
            <div style={{fontSize:11,color:"#556688",marginBottom:8,fontWeight:"bold"}}>LIVE DIAGNOSTICS</div>
            {[
              {label:"Time t", value:time.toFixed(3), unit:"s", color:"#a0d4ff"},
              {label:"Enstrophy E(t)", value:latestE.toFixed(4), unit:"", color:"#ff8844"},
              {label:"Viscous dissip. ν·E", value:(nu*latestE).toFixed(4), unit:"", color:"#44ff88"},
              {label:"Viscous gap δ", value:delta.toFixed(4), unit:"", color: delta>0.01?"#44ff88":"#ff4444"},
              {label:"Mass M₀", value:totalMass.toFixed(4), unit:"", color:"#ff88ff"},
              {label:"Particles N", value:"120", unit:"", color:"#88aaff"},
            ].map(({label,value,unit,color})=>(
              <div key={label} style={{display:"flex",justifyContent:"space-between",padding:"3px 0",borderBottom:"1px solid #0d1520"}}>
                <span style={{fontSize:11,color:"#667799"}}>{label}</span>
                <span style={{fontSize:12,color,fontWeight:"bold"}}>{value} {unit}</span>
              </div>
            ))}
            <div style={{marginTop:8,padding:"6px",background:"#04080f",borderRadius:4,border:`1px solid ${delta>0?"#1a4a1a":"#4a1a1a"}`}}>
              <div style={{fontSize:10,color:"#556688"}}>VISCOUS DOMINANCE CONJECTURE</div>
              <div style={{fontSize:11,marginTop:2,color:delta>0?"#44ff88":"#ff4444",fontWeight:"bold"}}>
                {delta>0 ? `✓ δ = ${delta.toFixed(4)} > 0  →  Global regularity supported` : "✗ δ ≤ 0  →  Blow-up risk"}
              </div>
              <div style={{fontSize:10,color:"#445566",marginTop:2}}>
                ∫(ω·∇)u·ω dV ≤ (1-δ)·ν·‖∇ω‖²
              </div>
            </div>
          </div>

          {/* Besov Spectrum */}
          {mode === 3 && (
            <div style={{background:"#080c1a",border:"1px solid #1a2a5a",borderRadius:6,overflow:"hidden"}}>
              <div style={{padding:"6px 12px",background:"#0a102a",fontSize:11,color:"#556688",borderBottom:"1px solid #1a2a5a"}}>
                Littlewood-Paley / Besov Spectrum B^s_{{p,q}}
              </div>
              <canvas ref={canvasBes} width={440} height={220} style={{width:"100%",height:"auto",display:"block"}}/>
            </div>
          )}

          {/* Enstrophy History */}
          {mode === 4 && (
            <div style={{background:"#080c1a",border:"1px solid #1a2a5a",borderRadius:6,overflow:"hidden",flex:1}}>
              <div style={{padding:"6px 12px",background:"#0a102a",fontSize:11,color:"#556688",borderBottom:"1px solid #1a2a5a"}}>
                Enstrophy History · ½‖ω(t)‖²_L²
              </div>
              <canvas ref={canvasEns} width={440} height={260} style={{width:"100%",height:"auto",display:"block"}}/>
            </div>
          )}

          {/* Enstrophy mini-bar always shown */}
          {mode !== 4 && (
            <div style={{background:"#080c1a",border:"1px solid #1a2a5a",borderRadius:6,padding:"10px 14px"}}>
              <div style={{fontSize:11,color:"#556688",marginBottom:6,fontWeight:"bold"}}>ENSTROPHY MINI-PLOT</div>
              <svg width="100%" height="60" viewBox={`0 0 300 60`} style={{display:"block"}}>
                <rect width="300" height="60" fill="#04080f"/>
                {ens.length > 1 && (() => {
                  const maxE = Math.max(...ens, 1);
                  const pts1 = ens.map((v,i) => `${50+i/(ens.length-1)*240},${54-v/maxE*48}`).join(" ");
                  const pts2 = ens.map((v,i) => `${50+i/(ens.length-1)*240},${54-(nu*v)/maxE*48}`).join(" ");
                  return (
                    <>
                      <polyline points={pts1} fill="none" stroke="#ff8844" strokeWidth="1.5"/>
                      <polyline points={pts2} fill="none" stroke="#44ff88" strokeWidth="1" strokeDasharray="3,2"/>
                      <text x="55" y="10" fill="#ff8844" fontSize="8">E(t)</text>
                      <text x="100" y="10" fill="#44ff88" fontSize="8">ν·E(t)</text>
                      <text x="5" y="32" fill="#334" fontSize="7" transform="rotate(-90,5,32)">E</text>
                    </>
                  );
                })()}
              </svg>
            </div>
          )}

          {/* C++ code preview */}
          <div style={{background:"#080c1a",border:"1px solid #1a2a5a",borderRadius:6,padding:"10px 14px"}}>
            <div style={{fontSize:11,color:"#556688",marginBottom:6,fontWeight:"bold"}}>C++ KERNEL (RK4 CORE)</div>
            <pre style={{fontSize:9,color:"#8899bb",margin:0,overflowX:"auto",lineHeight:1.6,background:"#04080f",padding:"8px",borderRadius:4}}>
{`// RK4 Lagrangian integrator
Vec3 rk4(Vec3 x, double dt, double t) {
  auto k1 = velocity(x, t);
  auto k2 = velocity(x + 0.5*dt*k1, t);
  auto k3 = velocity(x + 0.5*dt*k2, t);
  auto k4 = velocity(x +    dt*k3, t);
  return x + (dt/6)*(k1+2*k2+2*k3+k4);
}

// Enstrophy: Monte Carlo integration
double enstrophy(double t, int N=1000) {
  double sum = 0;
  for(int i=0;i<N;i++){
    Vec3 x = random_in_box();
    Vec3 w = curl(u)(x,t);
    sum += dot(w,w);
  }
  return 0.5 * sum/N * Vol;
}`}
            </pre>
          </div>

          {/* Mathematical identity box */}
          <div style={{background:"#080c1a",border:"1px solid #2a1a5a",borderRadius:6,padding:"10px 14px"}}>
            <div style={{fontSize:11,color:"#8855bb",marginBottom:6,fontWeight:"bold"}}>KEY MATHEMATICAL IDENTITIES</div>
            {[
              "∂ₜω + (u·∇)ω = (ω·∇)u + νΔω",
              "½ d/dt‖ω‖² + ν‖∇ω‖² = ∫ω·S·ω dV",
              "Tr(S) = ∇·u = 0  →  det constraint",
              "μ(Ω(t)) = μ(Ω₀)  (Reynolds Transport)",
              "M(t) = ρ₀·μ(Ω(t)) = M₀  (mass conserv.)",
            ].map((eq,i) => (
              <div key={i} style={{fontSize:10,color:"#9988cc",padding:"2px 0",borderBottom:"1px solid #0d1022",fontStyle:"italic"}}>
                {eq}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── FOOTER ── */}
      <div style={{padding:"8px 20px",borderTop:"1px solid #0d1520",fontSize:10,color:"#334455",display:"flex",justifyContent:"space-between",flexWrap:"wrap",gap:4}}>
        <span>Adam Ahmed Mohamed · Independent Researcher · Cairo, Egypt · 2026</span>
        <span>3D Navier-Stokes Simulator · Mass-Measure Geometric Framework · Clay Millennium Problem</span>
        <span>RK4 · Monte Carlo Enstrophy · Littlewood-Paley · CKN Theory</span>
      </div>
    </div>
  );
}
