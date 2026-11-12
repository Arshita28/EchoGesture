const assert = require('node:assert/strict');
const LetterVoice = require('../static/voice.js');
const said = [];
let cancelled = 0;
const synth = {cancel() { cancelled++; }, getVoices() { return [{name:'Samantha',lang:'en-US',localService:true}]; }, speak(u) { said.push(u.text); }};
class Utterance { constructor(text) { this.text = text; } }
const voice = new LetterVoice(synth, Utterance);
voice.observe('A','ok',0); assert.equal(said.length,0);
voice.enable(true);
voice.observe(null,'uncertain',0); assert.equal(said.length,0);
voice.observe('A','ok',0); voice.observe('A','ok',200); assert.deepEqual(said,['eigh']);
voice.observe(null,'uncertain',300); voice.observe('A','ok',500); assert.equal(said.length,2);
voice.observe(null,'no_hand',600); voice.observe(null,'no_hand',1700);
voice.observe('A','ok',1900); assert.deepEqual(said,['eigh','eigh','eigh']);
voice.observe('B','ok',2100); assert.deepEqual(said,['eigh','eigh','eigh','bee']);
voice.enable(false); voice.observe('C','ok',2300); assert.equal(said.length,4);
assert.ok(cancelled > 0);
const unsupported = new LetterVoice(null,null); unsupported.enable(true); assert.equal(unsupported.enabled,false);
console.log('Voice behavior checks passed');
let selected;
const femaleSynth = {cancel() {}, getVoices() { return [
  {name:'Alex',lang:'en-US',localService:true},
  {name:'Samantha',lang:'en-US',localService:true}
]; }, speak(u) { selected = u.voice.name; }};
const femaleVoice = new LetterVoice(femaleSynth, Utterance);
femaleVoice.enable(true); femaleVoice.observe('A','ok');
assert.equal(selected,'Samantha');
console.log('Female voice preference check passed');

const clips=[];
class FakeAudio { pause() {} play() { clips.push(this.src); return Promise.resolve(); } }
const repeatI = new LetterVoice(null,null,()=>{},FakeAudio);
repeatI.enable(true); repeatI.observe('I','ok'); repeatI.observe('I','ok');
repeatI.observe(null,'uncertain'); repeatI.observe('I','ok');
assert.equal(clips.filter(p=>p.includes('/I.wav')).length,2);
repeatI.observe(null,'no_hand'); repeatI.observe('I','ok');
assert.equal(clips.filter(p=>p.includes('/I.wav')).length,3);
console.log('I reappearance audio check passed');
const gVoice=new LetterVoice(null,null,()=>{},FakeAudio);
gVoice.enable(true);gVoice.observe('G','ok');
assert.ok(clips.at(-1).includes('/G.wav?'));
console.log('Displayed G selects the G audio file');
gVoice.observe('E','ok');assert.ok(clips.at(-1).includes('/E.wav?'));
console.log('Displayed E selects the E audio file');
let messageUtterance;const messageSynth={cancel(){},getVoices(){return [{name:'Samantha',lang:'en-US',localService:true}]},speak(u){messageUtterance=u;}};
const messageVoice=new LetterVoice(messageSynth,Utterance,()=>{},FakeAudio);
messageVoice.enable(true);assert.equal(messageVoice.speakMessage('HELLO WORLD'),true);
assert.equal(messageUtterance.text,'HELLO WORLD');const clipCount=clips.length;
messageVoice.observe('A','ok');assert.equal(clips.length,clipCount);
messageUtterance.onend();messageVoice.observe('A','ok');assert.equal(clips.length,clipCount+1);
messageVoice.speakMessage('SECOND');const oldEnd=messageUtterance.onend;messageVoice.reset();messageVoice.speakMessage('THIRD');oldEnd();assert.equal(messageVoice.messageSpeaking,true);
messageVoice.reset();assert.equal(messageVoice.messageSpeaking,false);
assert.equal(new LetterVoice(null,null).speakMessage('HELLO'),false);
console.log('Full-message speech, letter suppression, cancellation and stale completion checks passed');
