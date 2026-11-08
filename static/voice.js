/* Browser speech output for accepted, stable letters only. */
class LetterVoice {
  constructor(synth, Utterance, notify = () => {}, AudioType = typeof Audio !== 'undefined' ? Audio : null) {
    this.synth = synth; this.Utterance = Utterance; this.notify = notify;
    this.audio = AudioType ? new AudioType() : null;
    this.enabled = false; this.lastLetter = null; this.absentSince = null;
  }
  get supported() { return Boolean(this.audio || (this.synth && this.Utterance)); }
  reset() {
    if (this.synth) this.synth.cancel();
    if (this.audio) { this.audio.pause(); this.audio.currentTime = 0; }
    this.lastLetter = null; this.absentSince = null;
  }
  enable(value) {
    this.reset(); this.enabled = Boolean(value && this.supported);
    this.notify(this.enabled ? 'Voice on · Samantha (female)' : 'Voice off');
    if (this.enabled && this.audio) this.playClip('enabled');
  }
  playClip(name) {
    this.audio.pause();
    this.audio.src = `/static/audio/female/${name}.wav?v=2`;
    const playing = this.audio.play();
    if (playing && playing.catch) playing.catch(() => this.notify('Audio blocked — disable and enable voice again'));
  }
  observe(letter, status, now = Date.now()) {
    if (!this.enabled) return;
    if (status === 'no_hand') {
      if (this.absentSince === null) this.absentSince = now;
      if (now - this.absentSince >= 1000) this.lastLetter = null;
      return;
    }
    this.absentSince = null;
    if (!letter || letter === this.lastLetter) return;
    this.lastLetter = letter;
    if (this.audio) {
      if (/^[A-Z]$/.test(letter)) { this.playClip(letter); this.notify('Voice on · Samantha (female)'); }
      return;
    }
    this.synth.cancel(); // Replace old speech instead of building a queue.
    const utterance = new this.Utterance(['ay','bee','see','dee','ee','ef','jee','aitch','eye','jay','kay','el','em','en','oh','pee','cue','ar','ess','tee','you','vee','double you','ex','why','zee'][letter.charCodeAt(0)-65]);
    const voices = this.synth.getVoices();
    const english = voices.filter(v => /^en\b/i.test(v.lang));
    // The Web Speech API has no gender field. Prefer named female voices
    // when the browser exposes them; do not assume every Siri voice is female.
    const preferences = [/siri.*female|female.*siri/i, /samantha/i, /ava/i,
      /karen/i, /moira/i, /tessa/i, /victoria/i, /zira/i,
      /google.*female|female.*google/i, /female/i];
    let voice;
    for (const name of preferences) {
      voice = english.find(v => name.test(v.name));
      if (voice) break;
    }
    const preferred = Boolean(voice);
    if (!voice) { this.notify('Female voice unavailable — no default fallback used'); return; }
    this.notify(preferred ? `Voice on · ${voice.name}`
      : 'Voice on · System fallback (preferred female voice not available)');
    if (voice) utterance.voice = voice;
    utterance.lang = voice ? voice.lang : 'en-US';
    utterance.rate = .9;
    utterance.onerror = event => {
      if (!['interrupted', 'canceled'].includes(event.error))
        this.notify('Voice unavailable — try another browser or enable a system voice');
    };
    try { this.synth.speak(utterance); }
    catch (_) { this.notify('Voice unavailable in this browser'); }
  }
}
if (typeof window !== 'undefined') window.LetterVoice = LetterVoice;
if (typeof module !== 'undefined') module.exports = LetterVoice;
