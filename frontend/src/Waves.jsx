import { useEffect, useRef } from 'react';

const LAYERS = [
  { y: 78, amp: 62, len: 560, speed: 0.55, phase: 0.0, face: 'a', cls: 'one' },
  { y: 235, amp: 70, len: 640, speed: -0.45, phase: 1.4, face: 'b', cls: 'two' },
  { y: 394, amp: 58, len: 520, speed: 0.6, phase: 2.7, face: 'c', cls: 'three' },
  { y: 552, amp: 72, len: 680, speed: -0.5, phase: 0.8, face: 'b', cls: 'four' },
  { y: 710, amp: 60, len: 580, speed: 0.48, phase: 3.9, face: 'a', cls: 'five' },
  { y: 868, amp: 68, len: 620, speed: -0.58, phase: 5.1, face: 'b', cls: 'six' },
];
const X0 = -220, X1 = 1820, STEP = 20;

function wavePath(l, t) {
  const k = (Math.PI * 2) / l.len;
  let d = '';
  for (let x = X0; x <= X1; x += STEP) {
    const a = x * k + t * l.speed + l.phase;
    // two harmonics + slow breathing amplitude -> rolling, non-repeating swell
    const y = l.y + l.amp * (Math.sin(a) * 0.72 + Math.sin(a * 0.5 + t * 0.31) * 0.38 + Math.sin(a * 1.9 - t * 0.7) * 0.12) * (0.88 + 0.12 * Math.sin(t * 0.4 + l.phase));
    d += `${x === X0 ? 'M' : 'L'}${x} ${y.toFixed(1)} `;
  }
  return d;
}

export default function Waves() {
  const refs = useRef([]);
  useEffect(() => {
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    let raf, last = 0;
    const draw = (now) => {
      const t = now / 1000;
      refs.current.forEach((group, i) => {
        if (!group) return;
        const d = wavePath(LAYERS[i], t);
        for (const p of group.children) p.setAttribute('d', d);
      });
      if (!reduce && !document.hidden) raf = requestAnimationFrame(draw);
      else if (!reduce) raf = requestAnimationFrame(draw);
      last = now;
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <svg className="ocean-waves" viewBox="0 0 1600 1000" preserveAspectRatio="none">
    <defs>
      <linearGradient id="wave-face-a" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#278e83" /><stop offset="30%" stopColor="#69d4bd" /><stop offset="58%" stopColor="#278e83" /><stop offset="100%" stopColor="#143943" /></linearGradient>
      <linearGradient id="wave-face-b" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#0e2b35" /><stop offset="28%" stopColor="#278e83" /><stop offset="60%" stopColor="#143943" /><stop offset="100%" stopColor="#0e2b35" /></linearGradient>
      <linearGradient id="wave-face-c" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#143943" /><stop offset="32%" stopColor="#69d4bd" /><stop offset="64%" stopColor="#278e83" /><stop offset="100%" stopColor="#0e2b35" /></linearGradient>
      <linearGradient id="wave-ridge" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stopColor="#278e83" stopOpacity=".06" /><stop offset="45%" stopColor="#69d4bd" stopOpacity=".74" /><stop offset="100%" stopColor="#d8f7ec" stopOpacity=".12" /></linearGradient>
    </defs>
    {LAYERS.map((l, i) => <g key={l.cls} ref={(el) => (refs.current[i] = el)} className={`wave-layer wave-layer-${l.cls}`}>
      <path className="wave-shadow" />
      <path className="wave-body" />
      <path className="wave-ridge" />
    </g>)}
  </svg>;
}
