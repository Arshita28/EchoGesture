/* Browser speech output for accepted, stable letters only. */
class LetterVoice {
  constructor(synth, Utterance, notify = () => {}, AudioType = typeof Audio !== 'undefined' ? Audio : null) {
    this.synth = synth; this.Utterance = Utterance; this.notify = notify;
    this.audio = AudioType ? new AudioType() : null;
    this.enabled = false; this.lastLetter = null; this.messageSpeaking = false; this.speechJob = 0;
  }
  get supported() { return Boolean(this.audio || (this.synth && this.Utterance)); }
  reset() {
    this.speechJob++; this.messageSpeaking = false;
    if (this.synth) this.synth.cancel();
    if (this.audio) { this.audio.pause(); this.audio.currentTime = 0; }
    this.lastLetter = null;
  }
  enable(value) {
    this.reset(); this.enabled = Boolean(value && this.supported);
    this.notify(this.enabled ? 'Voice on' : 'Voice off');
    if (this.enabled && this.audio) this.playClip('enabled');
  }
  playClip(name) {
    this.audio.pause();
    this.audio.src = `/static/audio/female/${name}.wav?v=5`;
    const playing = this.audio.play();
    if (playing && playing.catch) playing.catch(() => this.notify('Audio blocked — disable and enable voice again'));
  }
  speakMessage(text) {
    const message = text.trim();
    if (!message) { this.notify('Add some text before speaking.'); return false; }
    if (!this.synth || !this.Utterance) { this.notify('Message speech is unavailable in this browser.'); return false; }
    const voices = this.synth.getVoices().filter(v => /^en\b/i.test(v.lang) && v.localService);
    const voice = [/samantha/i,/ava/i,/karen/i,/moira/i,/tessa/i,/victoria/i,/zira/i,/female/i]
      .map(pattern => voices.find(v => pattern.test(v.name))).find(Boolean);
    if (!voice) { this.notify('A local female English voice is unavailable. Your text is ready to copy.'); return false; }
    this.reset(); const job = this.speechJob;
    const utterance = new this.Utterance(message);
    utterance.voice = voice; utterance.lang = voice.lang; utterance.rate = .9;
    this.messageSpeaking = true;
    this.notify('Speaking');
    utterance.onend = () => { if(job !== this.speechJob) return; this.messageSpeaking = false; this.notify('Message spoken.'); };
    utterance.onerror = () => { if(job !== this.speechJob) return; this.messageSpeaking = false; this.notify('Message speech stopped or unavailable.'); };
    try { this.synth.speak(utterance); return true; }
    catch (_) { this.messageSpeaking = false; this.notify('Message speech is unavailable.'); return false; }
  }
  observe(letter, status) {
    if (!this.enabled || this.messageSpeaking) return;
    // Re-arm whenever the displayed recognition clears, even briefly.
    if (!letter) { this.lastLetter = null; return; }
    if (letter === this.lastLetter) return;
    this.lastLetter = letter;
    if (this.audio) {
      if (/^[A-Z]$/.test(letter)) { this.playClip(letter); this.notify('Voice on'); }
      return;
    }
    this.synth.cancel(); // Replace old speech instead of building a queue.
    const utterance = new this.Utterance(['eigh','bee','see','dee','e','ef','gee','aitch','eye','jay','kay','el','em','en','oh','pee','cue','ar','ess','tee','you','vee','double you','ex','why','zee'][letter.charCodeAt(0)-65]);
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
    this.notify('Voice on');
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
