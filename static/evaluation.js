const EVALUATION_LETTERS = 'ABCDEFGHIKLMNOPQRSTUVWXY'.split('');
/* Trial-level evaluation: frames within a trial are not independent examples. */
class EvaluationSession {
  constructor() { this.trials = []; this.active = null; }
  start(expected, cutoff, model) {
    if (this.active) throw Error('A trial is already recording');
    this.active = {expected, cutoff, model, started: new Date().toISOString(), frames: []};
  }
  observe(label, status, processingMs) {
    if (this.active) this.active.frames.push({label, status, processingMs});
  }
  cancel() { this.active = null; }
  finish() {
    const trial = this.active;
    this.active = null;
    if (!trial || trial.frames.length < 5) throw Error('Not enough predictions; retry the trial');
    const counts = {};
    for (const f of trial.frames) if (f.label) counts[f.label] = (counts[f.label] || 0) + 1;
    const winner = Object.keys(counts).sort((a,b) => counts[b]-counts[a])[0];
    trial.predicted = winner && counts[winner] > trial.frames.length / 2 ? winner : null;
    trial.outcome = trial.expected === 'NONE'
      ? (trial.predicted ? 'false_accept' : 'correct_rejection')
      : (trial.predicted === trial.expected ? 'correct' : trial.predicted ? 'wrong' : 'uncertain');
    this.trials.push(trial);
    return trial;
  }
  progress() {
    const counts = Object.fromEntries(EVALUATION_LETTERS.map(c => [c,0]));
    for (const t of this.trials) if (t.expected in counts) counts[t.expected]++;
    return {counts, tested:EVALUATION_LETTERS.filter(c=>counts[c]>0).length,
      repeated:EVALUATION_LETTERS.filter(c=>counts[c]>=3).length,
      untested:EVALUATION_LETTERS.filter(c=>counts[c]===0)};
  }
  nextLetter(after) {
    const remaining = new Set(this.progress().untested);
    const start = EVALUATION_LETTERS.indexOf(after);
    for (let i=1;i<=EVALUATION_LETTERS.length;i++) {
      const letter=EVALUATION_LETTERS[(start+i)%EVALUATION_LETTERS.length];
      if (remaining.has(letter)) return letter;
    }
    return null;
  }
  report() {
    const signs = this.trials.filter(t => t.expected !== 'NONE');
    const accepted = signs.filter(t => t.predicted);
    const correct = signs.filter(t => t.outcome === 'correct').length;
    const probes = this.trials.filter(t => t.expected === 'NONE');
    const perLetter = {};
    for (const t of signs) {
      const row = perLetter[t.expected] ||= {trials:0,correct:0,wrong:0,uncertain:0,predictions:{}};
      row.trials++; row[t.outcome]++;
      const label = t.predicted || 'UNCERTAIN';
      row.predictions[label] = (row.predictions[label] || 0) + 1;
    }
    const latencies = this.trials.flatMap(t => t.frames.map(f => f.processingMs)).filter(Number.isFinite).sort((a,b)=>a-b);
    return {created:new Date().toISOString(), protocol:'User-labeled three-second webcam trials. Majority of all sampled stabilized displays; no majority means abstain. Correlated frames are not independent tests. This session is not proof of signer independence. No images or training samples saved.',
      metrics:{signTrials:signs.length,correct,wrong:signs.filter(t=>t.outcome==='wrong').length,
        uncertain:signs.filter(t=>t.outcome==='uncertain').length,
        accuracy:signs.length ? correct/signs.length : null,
        coverage:signs.length ? accepted.length/signs.length : null,
        acceptedAccuracy:accepted.length ? correct/accepted.length : null,
        rejectionProbes:probes.length,falseAccepts:probes.filter(t=>t.outcome==='false_accept').length,
        medianServerMs:latencies.length ? latencies[Math.floor(latencies.length/2)] : null},
      progress:this.progress(),perLetter,trials:this.trials};
  }
}
function readableEvaluation(report) {
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const percent = value => value == null ? 'Not available (no applicable trials)' : `${(value*100).toFixed(1)}%`;
  const m = report.metrics;
  const rows = Object.entries(report.perLetter).map(([letter,r]) => `<tr><td>${escape(letter)}</td><td>${r.trials}</td><td>${r.correct}</td><td>${r.wrong}</td><td>${r.uncertain}</td></tr>`).join('');
  const trials = report.trials.map((t,i) => `<tr><td>${i+1}</td><td>${escape(t.expected)}</td><td>${escape(t.predicted || 'Uncertain — no stable majority')}</td><td>${escape(t.outcome.replaceAll('_',' '))}</td><td>${escape(t.cutoff)}%</td><td>${escape(t.model)}</td></tr>`).join('');
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ASL evaluation report</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#1c2920}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #cbd6ce;padding:10px;text-align:left}th{background:#eef4ef}.note{background:#f2f5ef;padding:16px}h1,h2{line-height:1.2}</style><h1>ASL webcam evaluation</h1><p>${escape(report.created)}</p><div class="note">${m.signTrials} letter trials: <strong>${m.correct} correct</strong>, ${m.wrong} wrong, ${m.uncertain} uncertain. An uncertain trial means no letter occupied more than half of the sampled stabilized displays. It does not mean a different letter was predicted.</div><p><strong>Alphabet coverage: ${report.progress.tested}/24 letters tested.</strong> ${report.progress.repeated}/24 have at least three trials. Untested: ${escape(report.progress.untested.join(', ') || 'none')}. Retest failures in fresh recordings before making accuracy claims.</p><h2>Summary</h2><ul><li>Letter accuracy (uncertain counts as unsuccessful): ${percent(m.accuracy)}</li><li>Accepted coverage: ${percent(m.coverage)}</li><li>Accuracy among accepted predictions: ${percent(m.acceptedAccuracy)}</li><li>Rejection checks: ${m.rejectionProbes}; false accepts: ${m.falseAccepts}</li><li>Median measured server time: ${m.medianServerMs == null ? 'Not available' : m.medianServerMs+' ms'}</li></ul><h2>Results by letter</h2><table><tr><th>Letter</th><th>Trials</th><th>Correct</th><th>Wrong</th><th>Uncertain</th></tr>${rows}</table><h2>Individual trials</h2><table><tr><th>Trial</th><th>Expected</th><th>Prediction</th><th>Outcome</th><th>Cutoff</th><th>Model</th></tr>${trials}</table><p class="note">${escape(report.protocol)} One trial is one test; repeated frames are not separate accuracy examples. A small session does not establish overall project accuracy.</p></html>`;
}
function mountEvaluation({ready,cutoff,model}) {
  const session = new EvaluationSession();
  const get = id => document.getElementById(id);
  let timer = null;
  function controls(recording) {
    get('evalStart').disabled = recording;
    get('evalExpected').disabled = recording;
    // Acceptance is now a fixed product setting, not a visible control.
  }
  function cancel() {
    if (!session.active) return;
    clearTimeout(timer); session.cancel(); controls(false);
    get('evalStatus').textContent = 'Trial cancelled; no result recorded.';
  }
  get('evalStart').onclick = () => {
    if (!ready()) { get('evalStatus').textContent = 'Start the webcam with a trained model first. Evaluation uses live webcam trials.'; return; }
    session.start(get('evalExpected').value, Number(cutoff()), model());
    controls(true);
    get('evalStatus').textContent = 'Recording for 3 seconds — hold the chosen sign steady.';
    timer = setTimeout(() => {
      try {
        const trial = session.finish(), r = session.report();
        get('evalStatus').textContent = `${session.trials.length} trials · ${r.metrics.correct} correct letters · ${r.metrics.wrong} wrong · ${r.metrics.uncertain} uncertain · ${r.metrics.falseAccepts} false accepts. Last: ${trial.expected} → ${trial.predicted || 'uncertain'} (${trial.outcome.replaceAll('_',' ')}).`;
        get('evalDownload').disabled = false;
        const progress = session.progress();
        get('evalProgress').textContent = `Alphabet coverage: ${progress.tested}/24 tested · ${progress.repeated}/24 with at least three trials. Untested: ${progress.untested.join(', ') || 'none'}.`;
        const failures=Object.entries(r.perLetter).filter(([,v])=>v.wrong || v.uncertain).map(([c])=>c);
        get('evalFocus').textContent = failures.length ? `Retest these letters: ${failures.join(', ')}. Use a new pose or recording; keep this report separate from training.` : 'No failed letter trials recorded yet. Repeat each letter in fresh poses; a single success is not enough.';
        if (trial.expected !== 'NONE') {
          const next=session.nextLetter(trial.expected);
          if (next) { get('evalExpected').value=next; get('evalStatus').textContent += ` Next expected: ${next}. Make that sign, then record when ready.`; }
          else get('evalStatus').textContent += ' All 24 letters tested. Repeat weak letters and add rejection checks.';
        }
      } catch (error) { get('evalStatus').textContent = error.message; }
      controls(false);
    }, 3000);
  };
  function download(content,type,filename) {
    const url=URL.createObjectURL(new Blob([content],{type})),link=document.createElement('a');
    link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  get('evalDownload').onclick = () => download(readableEvaluation(session.report()),'text/html','asl-webcam-evaluation.html');
  return {get active(){return session.active!==null;},cancel, observe:(label,result)=>session.observe(label,result.status,result.processing_ms)};
}
if (typeof window !== 'undefined') window.mountEvaluation = mountEvaluation;
if (typeof module !== 'undefined') module.exports = EvaluationSession;
if (typeof module !== 'undefined') module.exports.readableEvaluation = readableEvaluation;
