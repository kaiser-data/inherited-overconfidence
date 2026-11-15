import React from 'react';
import {
  AbsoluteFill, Audio, Easing, interpolate, Sequence, staticFile, useCurrentFrame,
  continueRender, delayRender,
} from 'remotion';
import manifest from './manifest.json';

export const FPS = 30;
const XF = 14; // crossfade frames between scenes
const LEAD = 0.4; // seconds of picture before the voice starts
const TAIL = 0.7; // seconds after the voice ends

// palette of the slide deck (dark)
const C = {
  ground: '#141A20', tone: '#212A32', line: '#3A454F', ink: '#E7E9E5', mute: '#A6B0B8',
  acc: '#F0894A', right: '#74BAC6', wrong: '#E0685A',
};
const SERIF = '"Plex Serif", "EB Garamond", Georgia, serif';
const SANS = '"Plex Sans", Inter, "DejaVu Sans", sans-serif';
const MONO = '"Plex Mono", "DejaVu Sans Mono", monospace';

// fonts: IBM Plex (OFL), shipped in public/fonts
const FONTS: [string, string, string][] = [
  ['Plex Serif', '600', 'fonts/plex-serif-600.woff2'], ['Plex Sans', '400', 'fonts/plex-sans-400.woff2'],
  ['Plex Sans', '600', 'fonts/plex-sans-600.woff2'], ['Plex Mono', '500', 'fonts/plex-mono-500.woff2'],
];
if (typeof document !== 'undefined') {
  const h = delayRender('fonts');
  Promise.all(FONTS.map(([fam, weight, file]) =>
    new FontFace(fam, `url(${staticFile(file)}) format("woff2")`, { weight }).load().then((f) => document.fonts.add(f))))
    .then(() => continueRender(h)).catch(() => continueRender(h));
}

type Line = { file: string; dur: number; text: string };
const LINES = (manifest as unknown as { lines: Record<string, Line> }).lines;
const DISCLOSURE = (manifest as unknown as { disclosure: string }).disclosure;

/* ---------- helpers ---------- */
const clamp = { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' } as const;
const out = Easing.out(Easing.cubic);
const io = Easing.inOut(Easing.cubic);
/** value moving from a to b between progress p0 and p1 of the scene */
const tw = (p: number, p0: number, p1: number, a = 0, b = 1, ease = out) =>
  interpolate(p, [p0, p1], [a, b], { ...clamp, easing: ease });
const useP = (dur: number) => { const f = useCurrentFrame(); return { f, p: f / dur }; };

const Appear: React.FC<{ p: number; at: number; len?: number; y?: number; style?: React.CSSProperties; children: React.ReactNode }> =
  ({ p, at, len = 0.06, y = 26, style, children }) => (
    <div style={{ opacity: tw(p, at, at + len), transform: `translateY(${tw(p, at, at + len, y, 0)}px)`, ...style }}>{children}</div>
  );

const H: React.FC<{ children: React.ReactNode; size?: number; style?: React.CSSProperties }> = ({ children, size = 64, style }) => (
  <div style={{ fontFamily: SERIF, fontWeight: 600, fontSize: size, lineHeight: 1.12, color: C.ink, letterSpacing: '-0.01em', ...style }}>{children}</div>
);
const Num: React.FC<{ children: React.ReactNode; color?: string; size?: number }> = ({ children, color = C.ink, size }) => (
  <span style={{ fontFamily: MONO, fontWeight: 500, color, fontSize: size, fontVariantNumeric: 'tabular-nums' }}>{children}</span>
);
const Frame: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill style={{ background: C.ground, color: C.ink, fontFamily: SANS, fontSize: 34, padding: '70px 150px 190px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>{children}</AbsoluteFill>
);

/** the reply card of the student: question, answer, confidence */
const Reply: React.FC<{ qChars: number; showAnswer: number; conf: number; confColor: string; showConf: number; scale?: number }> =
  ({ qChars, showAnswer, conf, confColor, showConf, scale = 1 }) => {
    const q = 'Who wrote the opera A Masked Ball?';
    return (
      <div style={{ transform: `scale(${scale})`, transformOrigin: '0 0', width: 1180 }}>
        <div style={{ fontSize: 30, color: C.mute, marginBottom: 14 }}>Question</div>
        <H size={62} style={{ minHeight: 76 }}>{q.slice(0, qChars)}<span style={{ opacity: qChars < q.length ? 1 : 0, color: C.acc }}>|</span></H>
        <div style={{ marginTop: 44, background: C.tone, borderRadius: 6, padding: '38px 46px', opacity: showAnswer,
          transform: `translateY(${(1 - showAnswer) * 30}px)` }}>
          <div style={{ fontSize: 28, color: C.mute, marginBottom: 12 }}>Small model</div>
          <div style={{ fontFamily: MONO, fontWeight: 500, fontSize: 48, lineHeight: 1.5 }}>
            <div>Answer: Giacomo Puccini</div>
            <div style={{ opacity: showConf }}>Confidence: <span style={{ color: confColor, fontSize: 78 }}>{Math.round(conf)}%</span></div>
          </div>
        </div>
      </div>
    );
  };

/** a horizontal slider with stops; pos 0..1 */
const Slider: React.FC<{ pos: number; stops: string[]; active: number; label: string; width?: number }> = ({ pos, stops, active, label, width = 760 }) => (
  <div style={{ width }}>
    <div style={{ fontWeight: 600, fontSize: 30, marginBottom: 30 }}>{label}</div>
    <div style={{ position: 'relative', height: 14, background: C.line, borderRadius: 14 }}>
      <div style={{ position: 'absolute', left: 0, width: `${pos * 100}%`, top: 0, bottom: 0, background: C.acc, borderRadius: 14, opacity: 0.55 }} />
      <div style={{ position: 'absolute', left: `${pos * 100}%`, top: '50%', width: 54, height: 54, margin: '-27px 0 0 -27px', borderRadius: 54,
        background: C.acc, border: `8px solid ${C.ground}`, boxShadow: '0 6px 18px rgba(0,0,0,.45)' }} />
    </div>
    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 30, fontSize: 25, color: C.mute }}>
      {stops.map((s, i) => <span key={s} style={{ color: i === active ? C.ink : C.mute, fontWeight: i === active ? 600 : 400, width: `${100 / stops.length}%`,
        textAlign: i === 0 ? 'left' : i === stops.length - 1 ? 'right' : 'center' }}>{s}</span>)}
    </div>
  </div>
);

const Bar: React.FC<{ label: string; value: number; color: string; width?: number; max?: number }> = ({ label, value, color, width = 760, max = 100 }) => (
  <div style={{ display: 'grid', gridTemplateColumns: `170px ${width - 320}px 150px`, alignItems: 'center', gap: 22, marginTop: 16 }}>
    <span style={{ fontSize: 28, color: C.mute }}>{label}</span>
    <span style={{ height: 36, background: C.tone, borderRadius: 5, overflow: 'hidden', display: 'block' }}>
      <span style={{ display: 'block', height: '100%', width: `${(value / max) * 100}%`, background: color, borderRadius: 5 }} />
    </span>
    <Num size={44} color={C.ink}>{Math.round(value)}%</Num>
  </div>
);

/* ---------- scenes ---------- */

const Hook: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  const wrong = tw(p, 0.7, 0.76);
  return (
    <Frame>
      <Reply qChars={Math.round(tw(p, 0.02, 0.2, 0, 34, Easing.linear))} showAnswer={tw(p, 0.27, 0.33)} showConf={tw(p, 0.4, 0.44)}
        conf={tw(p, 0.42, 0.58, 0, 99)} confColor={C.acc} scale={1.15} />
      <div style={{ position: 'absolute', left: 1090, top: 575, opacity: wrong, transform: `rotate(-4deg) scale(${tw(p, 0.7, 0.76, 1.5, 1)})`,
        border: `6px solid ${C.wrong}`, color: C.wrong, borderRadius: 8, padding: '18px 34px', fontWeight: 600, fontSize: 60 }}>Wrong</div>
      <Appear p={p} at={0.82} style={{ position: 'absolute', left: 150, top: 835, fontSize: 40, color: C.mute }}>
        The composer was <span style={{ color: C.ink, fontWeight: 600 }}>Giuseppe Verdi</span>.
      </Appear>
    </Frame>
  );
};

const Flip: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  const move = tw(p, 0.42, 0.62, 0, 1, io);
  const conf = 99 - 59 * move;
  const col = move > 0.55 ? C.right : C.acc;
  return (
    <Frame>
      <Reply qChars={34} showAnswer={1} showConf={1} conf={conf} confColor={col} scale={0.78} />
      <div style={{ position: 'absolute', right: 150, top: 250, width: 620 }}>
        <Appear p={p} at={0.2}>
          <Slider pos={1 - move} stops={['fitted to the student', "the teacher's own"]} active={move > 0.5 ? 0 : 1}
            label="Confidence written in the training data" width={620} />
        </Appear>
        <div style={{ marginTop: 70, fontSize: 32, lineHeight: 1.7 }}>
          {['Same model', 'Same training answers', 'Same wrong answer'].map((t, i) => (
            <Appear key={t} p={p} at={0.05 + i * 0.06} y={14}><span style={{ color: C.right, fontWeight: 600 }}>= </span>{t}</Appear>
          ))}
          <Appear p={p} at={0.64} y={14} style={{ marginTop: 22 }}><span style={{ color: C.acc, fontWeight: 600 }}>≠ </span>
            <span style={{ fontWeight: 600 }}>How sure it sounds</span></Appear>
        </div>
      </div>
    </Frame>
  );
};

const Node: React.FC<{ r: number; title: string; sub: string; color: string }> = ({ r, title, sub, color }) => (
  <div style={{ textAlign: 'center', width: 440 }}>
    <div style={{ width: r, height: r, borderRadius: r, background: color, margin: `${(300 - r) / 2}px auto` }} />
    <div style={{ fontWeight: 600, fontSize: 38 }}>{title}</div>
    <div style={{ color: C.mute, fontSize: 29, marginTop: 6 }}>{sub}</div>
  </div>
);

const Question: React.FC<{ dur: number }> = ({ dur }) => {
  const { f, p } = useP(dur);
  return (
    <Frame>
      <Appear p={p} at={0.02}><H>Where does a small model's confidence come from?</H></Appear>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginTop: 70, position: 'relative' }}>
        <Appear p={p} at={0.1}><Node r={280} title="Large model" sub="the teacher writes the answers" color={C.line} /></Appear>
        <Appear p={p} at={0.3}><Node r={150} title="Small model" sub="the student is trained on them" color={C.line} /></Appear>
        {[0, 1].map((i) => {
          const t = ((f + i * 80) % 160) / 160; // cards flying from teacher to student
          return (
            <div key={i} style={{ position: 'absolute', left: 430 + t * 640, top: 96 + Math.sin(t * Math.PI) * -34, opacity: tw(p, 0.22, 0.3) * Math.sin(t * Math.PI),
              background: C.tone, borderRadius: 5, padding: '12px 20px', fontFamily: MONO, fontSize: 24, lineHeight: 1.45, whiteSpace: 'nowrap' }}>
              <div>Answer: …</div><div>Confidence: <span style={{ color: C.acc }}>95%</span></div>
            </div>
          );
        })}
      </div>
      <Appear p={p} at={0.55} style={{ marginTop: 50, fontSize: 38, color: C.mute }}>
        The teacher's <span style={{ color: C.acc, fontWeight: 600 }}>confidence</span> travels with its answers. What does the student do with it?
      </Appear>
    </Frame>
  );
};

const Design: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  const cards: [string, string, string][] = [["the teacher's own", '99%', C.acc], ['calibrated for the teacher', '90%', C.acc], ['fitted to the student', '40%', C.right], ['shuffled', '?%', C.mute]];
  const glow = tw(p, 0.62, 0.72);
  return (
    <Frame>
      <Appear p={p} at={0.02}><H>Same answers, four confidence settings</H></Appear>
      <Appear p={p} at={0.08} style={{ marginTop: 26, fontSize: 32, color: C.mute }}>
        Teacher: DeepSeek-V4.1-Flash, 90% right.&nbsp;&nbsp;Students: Qwen3.5-9B and Granite-4.1-8B, about 60% right.
      </Appear>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 30, marginTop: 70 }}>
        {cards.map(([name, conf, col], i) => (
          <Appear key={name} p={p} at={0.16 + i * 0.07}>
            <div style={{ background: C.tone, borderRadius: 6, padding: '30px 30px 34px' }}>
              <div style={{ fontSize: 26, color: C.mute, minHeight: 70 }}>{name}</div>
              <div style={{ fontFamily: MONO, fontWeight: 500, fontSize: 30, lineHeight: 1.7, marginTop: 16 }}>
                <div style={{ opacity: 1 - 0.45 * glow }}>Answer: Puccini</div>
                <div>Confidence: <span style={{ color: col, fontSize: 30 + 16 * glow }}>{conf}</span></div>
              </div>
            </div>
          </Appear>
        ))}
      </div>
      <Appear p={p} at={0.5} style={{ marginTop: 56, fontSize: 36 }}>
        The answers stay <span style={{ fontWeight: 600 }}>identical</span>. Only the <span style={{ color: C.acc, fontWeight: 600 }}>confidence</span> is changed.
      </Appear>
      <Appear p={p} at={0.8} style={{ marginTop: 14, fontSize: 28, color: C.mute }}>The student's reply to the opera question, after training on each setting. Predictions were committed before the runs.</Appear>
    </Frame>
  );
};

const ResultHead: React.FC<{ p: number; n: string; children: React.ReactNode }> = ({ p, n, children }) => (
  <Appear p={p} at={0.02}>
    <div style={{ display: 'flex', gap: 30, alignItems: 'baseline' }}>
      <span style={{ fontFamily: MONO, fontWeight: 500, fontSize: 60, color: C.acc }}>{n}</span><H size={60}>{children}</H>
    </div>
  </Appear>
);

const Result1: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  // Table A, Qwen3.5-9B, sampled answer: fitted 55/60, calibrated teacher 89/57, teacher's own 98/58
  const pos = interpolate(p, [0.5, 0.62, 0.68, 0.8, 1], [0, 0.5, 0.5, 1, 1], { ...clamp, easing: io });
  const says = interpolate(pos, [0, 0.5, 1], [55, 89, 98]);
  const right = interpolate(pos, [0, 0.5, 1], [60, 57, 58]);
  const gap = says - right;
  return (
    <Frame>
      <ResultHead p={p} n="1">Confidence can be set without changing knowledge</ResultHead>
      <div style={{ display: 'flex', gap: 110, marginTop: 80 }}>
        <Appear p={p} at={0.14}>
          <Slider pos={pos} stops={['fitted to the student', 'calibrated teacher', "teacher's own"]} active={pos < 0.25 ? 0 : pos < 0.75 ? 1 : 2}
            label="Confidence written in the training data" />
          <div style={{ marginTop: 64, fontWeight: 600, fontSize: 30 }}>Student Qwen3.5-9B on 1,000 unseen questions</div>
          <Bar label="says" value={says} color={C.acc} />
          <Bar label="is right" value={right} color={C.right} />
        </Appear>
        <Appear p={p} at={0.22} style={{ paddingTop: 20 }}>
          <div style={{ fontSize: 32, color: C.mute }}>Overconfidence gap</div>
          <div style={{ fontFamily: MONO, fontWeight: 500, fontSize: 170, lineHeight: 1.1, color: gap > 2 ? C.acc : C.right }}>
            {gap >= 0 ? '+' : '−'}{Math.abs(Math.round(gap))}
          </div>
          <div style={{ fontSize: 30, color: C.mute, width: 520, lineHeight: 1.4 }}>says minus is right, in points. Zero is the target.</div>
        </Appear>
      </div>
    </Frame>
  );
};

const GapRow: React.FC<{ p: number; at: number; label: string; q: number; g: number }> = ({ p, at, label, q, g }) => {
  const grow = tw(p, at, at + 0.12);
  const zero = 12; // percent of the rail where 0 sits; one point = 2 percent
  const w = Math.abs(q) * 2 * grow;
  const col = q > 0 ? C.acc : C.right;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '470px 760px 330px', gap: 30, alignItems: 'center', marginTop: 34, opacity: tw(p, at - 0.03, at) }}>
      <span style={{ fontWeight: 600, fontSize: 36 }}>{label}</span>
      <span style={{ position: 'relative', height: 60, display: 'block' }}>
        <span style={{ position: 'absolute', left: `${zero}%`, top: -14, bottom: -14, width: 3, background: C.line }} />
        <span style={{ position: 'absolute', top: 0, bottom: 0, left: `${q > 0 ? zero : zero - w}%`, width: `${w}%`, background: col, borderRadius: 5 }} />
      </span>
      <Num size={52} color={col}>{q > 0 ? '+' : '−'}{Math.round(Math.abs(q) * grow)} / {g > 0 ? '+' : '−'}{Math.round(Math.abs(g) * grow)}</Num>
    </div>
  );
};

const Result2: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  return (
    <Frame>
      <ResultHead p={p} n="2">The right setting belongs to the student, not the teacher</ResultHead>
      <div style={{ marginTop: 56 }}>
        <GapRow p={p} at={0.3} label="Teacher's own confidence" q={40} g={38} />
        <GapRow p={p} at={0.5} label="Teacher calibrated first" q={31} g={27} />
        <GapRow p={p} at={0.8} label="Fitted to the student" q={-5} g={-6} />
      </div>
      <Appear p={p} at={0.34} style={{ marginTop: 40, fontSize: 28, color: C.mute }}>Overconfidence gap in points: Qwen3.5-9B / Granite-4.1-8B. Zero is the target.</Appear>
      <Appear p={p} at={0.22} style={{ marginTop: 34, fontSize: 36, background: C.tone, borderRadius: 6, padding: '26px 34px', width: 1500 }}>
        The student takes over the teacher's <span style={{ color: C.acc, fontWeight: 600 }}>certainty</span>, but not its <span style={{ color: C.right, fontWeight: 600 }}>ability</span>. "90%" is honest for a teacher that is right nine times in ten; the student is right six times in ten.
      </Appear>
    </Frame>
  );
};

const Dots: React.FC<{ p: number; at: number; right: number; title: string; pct: string; color: string }> = ({ p, at, right, title, pct, color }) => (
  <div style={{ opacity: tw(p, at, at + 0.05), width: 660 }}>
    <div style={{ fontWeight: 600, fontSize: 34 }}>{title}</div>
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 46px)', gap: 12, marginTop: 28 }}>
      {Array.from({ length: 40 }).map((_, i) => {
        const on = tw(p, at + 0.04 + i * 0.004, at + 0.08 + i * 0.004);
        return <span key={i} style={{ width: 46, height: 46, borderRadius: 46, background: i < right ? color : C.line, transform: `scale(${on})`, display: 'block' }} />;
      })}
    </div>
    <div style={{ marginTop: 22, fontSize: 30, color: C.mute }}>right on its more confident half</div>
    <Num size={72} color={color}>{pct}</Num>
  </div>
);

const Result3: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  return (
    <Frame>
      <ResultHead p={p} n="3">The teacher's confidence still teaches which questions are hard</ResultHead>
      <div style={{ display: 'flex', gap: 150, marginTop: 44 }}>
        <Dots p={p} at={0.3} right={30} title="Trained on the teacher's confidence" pct="75% / 80%" color={C.right} />
        <Dots p={p} at={0.68} right={24} title="Same numbers, shuffled" pct="60% / 62%" color={C.mute} />
      </div>
      <Appear p={p} at={0.5} style={{ marginTop: 26, fontSize: 26, color: C.mute }}>
        Dots: 40 answers from the student's more confident half; filled = right (Qwen3.5-9B). Numbers: Qwen3.5-9B / Granite-4.1-8B.
      </Appear>
    </Frame>
  );
};

const Extend: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  // Qwen3.5-9B, calibrated teacher, sampled answer: trivia 89 / 57, Natural Questions 80 / 23 (results/ext_stats.json)
  const m = tw(p, 0.66, 0.84, 0, 1, io);
  return (
    <Frame>
      <Appear p={p} at={0.02}><H>Two more tests</H></Appear>
      <div style={{ display: 'flex', gap: 110, marginTop: 60 }}>
        <Appear p={p} at={0.1} style={{ width: 720 }}>
          <div style={{ fontWeight: 600, fontSize: 36 }}>Confidence in words, not numbers</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginTop: 26 }}>
            {['almost certain', 'very confident', 'fairly confident', 'unsure', 'doubtful', 'guessing'].map((w, i) => (
              <span key={w} style={{ background: C.tone, borderRadius: 5, padding: '10px 18px', fontFamily: MONO, fontSize: 26, opacity: tw(p, 0.12 + i * 0.02, 0.16 + i * 0.02) }}>{w}</span>
            ))}
          </div>
          <Appear p={p} at={0.3} style={{ marginTop: 40, fontSize: 32, lineHeight: 1.6 }}>
            <div>Calibrated teacher: <Num color={C.acc} size={44}>+32 / +28</Num></div>
            <div>Fitted to the student: <Num color={C.right} size={44}>−2 / −2</Num></div>
            <div style={{ color: C.mute, fontSize: 28 }}>The same result as with numbers.</div>
          </Appear>
        </Appear>
        <Appear p={p} at={0.46} style={{ width: 780 }}>
          <div style={{ fontWeight: 600, fontSize: 36 }}>New, harder questions</div>
          <div style={{ marginTop: 26, fontSize: 30, color: C.mute }}>{m < 0.5 ? 'Trivia, like the training questions' : 'Natural Questions, never seen in training'}</div>
          <Bar label="says" value={89 - 9 * m} color={C.acc} width={780} />
          <Bar label="is right" value={57 - 34 * m} color={C.right} width={780} />
          <div style={{ marginTop: 34, fontSize: 32 }}>Overconfidence gap <Num color={C.acc} size={64}>+{Math.round(32 + 24 * m)}</Num></div>
          <div style={{ color: C.mute, fontSize: 26, marginTop: 6 }}>Qwen3.5-9B trained on the calibrated teacher.</div>
        </Appear>
      </div>
    </Frame>
  );
};

const Meaning: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  const words = 'A stated confidence tells us first what the trainer put in, and only then what the model knows.'.split(' ');
  return (
    <Frame>
      <div style={{ fontFamily: SERIF, fontWeight: 600, fontSize: 84, lineHeight: 1.22, width: 1500, marginTop: 40 }}>
        {words.map((w, i) => {
          const at = 0.03 + (i / words.length) * 0.55;
          const hot = ['trainer', 'put', 'in,'].includes(w);
          return <span key={i} style={{ opacity: tw(p, at, at + 0.05, 0.12, 1), color: hot ? C.acc : C.ink }}>{w} </span>;
        })}
      </div>
      <div style={{ display: 'flex', gap: 40, marginTop: 70, fontSize: 44, fontWeight: 600 }}>
        <Appear p={p} at={0.68}><span style={{ color: C.right }}>A design tool</span></Appear>
        <Appear p={p} at={0.82}><span style={{ color: C.acc }}>and a risk</span></Appear>
      </div>
    </Frame>
  );
};

const Cta: React.FC<{ dur: number }> = ({ dur }) => {
  const { p } = useP(dur);
  return (
    <Frame>
      <Appear p={p} at={0.02}><H size={84}>A Confidence Slider<br />for Small LLMs</H></Appear>
      <Appear p={p} at={0.1} style={{ marginTop: 26, fontSize: 40, color: C.mute }}>The Dunning–Kruger risk in model distillation.</Appear>
      <Appear p={p} at={0.25} style={{ marginTop: 60, fontSize: 34 }}>
        A first step: one large teacher, two students, two question sets.
      </Appear>
      <Appear p={p} at={0.5} style={{ marginTop: 40 }}>
        <div style={{ fontSize: 30, color: C.mute }}>Paper, code, data and protocol</div>
        <div style={{ fontFamily: MONO, fontWeight: 500, fontSize: 44, color: C.acc, marginTop: 8 }}>github.com/kaiser-data/inherited-overconfidence</div>
      </Appear>
      <Appear p={p} at={0.6} style={{ marginTop: 44, fontSize: 28, color: C.mute }}>
        Martin Kaiser, with Apart Research. AI x Epistemics Research Sprint, November 2026.
      </Appear>
      <div style={{ position: 'absolute', left: 150, right: 150, bottom: 90, fontSize: 22, color: C.mute }}>{DISCLOSURE}</div>
    </Frame>
  );
};

/* ---------- assembly ---------- */

type Scene = { id: string; C: React.FC<{ dur: number }>; captions?: boolean };
const SCENES: Scene[] = [
  { id: '01-hook', C: Hook }, { id: '02-flip', C: Flip }, { id: '03-question', C: Question }, { id: '04-design', C: Design },
  { id: '05-result1', C: Result1 }, { id: '06-result2', C: Result2 }, { id: '07-result3', C: Result3 }, { id: '07b-new', C: Extend },
  { id: '08-meaning', C: Meaning }, { id: '09-cta', C: Cta, captions: false },
];
const sceneFrames = (s: Scene) => Math.round((LEAD + LINES[s.id].dur + TAIL) * FPS) + XF;
const starts = () => { let t = 0; return SCENES.map((s) => { const at = t; t += sceneFrames(s) - XF; return at; }); };
export const totalFrames = () => { const st = starts(); return st[st.length - 1] + sceneFrames(SCENES[SCENES.length - 1]); };

const Fade: React.FC<{ dur: number; first: boolean; last: boolean; children: React.ReactNode }> = ({ dur, first, last, children }) => {
  const f = useCurrentFrame();
  const o = interpolate(f, [0, XF, dur - XF, dur], [first ? 1 : 0, 1, 1, last ? 1 : 0], clamp);
  return <AbsoluteFill style={{ opacity: o }}>{children}</AbsoluteFill>;
};

/** narration as captions: one sentence at a time, timed by its share of the words */
const Captions: React.FC<{ text: string; voiceFrames: number }> = ({ text, voiceFrames }) => {
  const f = useCurrentFrame() - LEAD * FPS;
  const parts = text.match(/[^.?!]+[.?!]+/g) ?? [text];
  const total = text.split(/\s+/).length;
  let acc = 0;
  let cur = '';
  for (const s of parts) {
    const n = s.trim().split(/\s+/).length;
    if (f >= (acc / total) * voiceFrames - 4 && f < ((acc + n) / total) * voiceFrames + 6) cur = s.trim();
    acc += n;
  }
  if (!cur) return null;
  return (
    <div style={{ position: 'absolute', left: 150, right: 150, bottom: 84, textAlign: 'center', fontFamily: SANS, fontSize: 34, lineHeight: 1.35, color: C.ink }}>
      <span style={{ background: 'rgba(11,15,19,0.82)', padding: '8px 18px', borderRadius: 6, boxDecorationBreak: 'clone', WebkitBoxDecorationBreak: 'clone' }}>{cur}</span>
    </div>
  );
};

/** the film's own progress as a slider at the bottom */
const Track: React.FC = () => {
  const f = useCurrentFrame();
  const st = starts();
  const total = totalFrames();
  return (
    <div style={{ position: 'absolute', left: 150, right: 150, bottom: 44, height: 20 }}>
      <div style={{ position: 'absolute', left: 0, right: 0, top: 9, height: 2, background: C.line }} />
      {st.map((s, i) => <div key={i} style={{ position: 'absolute', left: `${(s / total) * 100}%`, top: 4, width: 2, height: 12, background: C.line }} />)}
      <div style={{ position: 'absolute', left: `${(f / total) * 100}%`, top: 10, width: 20, height: 20, margin: '-10px 0 0 -10px', borderRadius: 20,
        background: C.acc, border: `4px solid ${C.ground}` }} />
    </div>
  );
};

export const Promo: React.FC = () => {
  const st = starts();
  return (
    <AbsoluteFill style={{ background: C.ground }}>
      {SCENES.map((s, i) => {
        const dur = sceneFrames(s);
        const voice = Math.round(LINES[s.id].dur * FPS);
        return (
          <Sequence key={s.id} from={st[i]} durationInFrames={dur}>
            <Fade dur={dur} first={i === 0} last={i === SCENES.length - 1}>
              <s.C dur={dur} />
              {s.captions === false ? null : <Captions text={LINES[s.id].text} voiceFrames={voice} />}
            </Fade>
            <Sequence from={Math.round(LEAD * FPS)}><Audio src={staticFile(LINES[s.id].file)} /></Sequence>
          </Sequence>
        );
      })}
      <Track />
    </AbsoluteFill>
  );
};
