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
voice.observe('A','ok',0); voice.observe('A','ok',200); assert.deepEqual(said,['ay']);
voice.observe(null,'uncertain',300); voice.observe('A','ok',500); assert.equal(said.length,1);
voice.observe(null,'no_hand',600); voice.observe(null,'no_hand',1700);
voice.observe('A','ok',1900); assert.deepEqual(said,['ay','ay']);
voice.observe('B','ok',2100); assert.deepEqual(said,['ay','ay','bee']);
voice.enable(false); voice.observe('C','ok',2300); assert.equal(said.length,3);
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
