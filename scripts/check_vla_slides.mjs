// Start headless Chrome with --remote-debugging-port=9247 and a temporary --user-data-dir.
// Run: node scripts/check_vla_slides.mjs [HTML path] [debugging port] [optional preview directory]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const file = path.resolve(process.argv[2] || fileURLToPath(new URL('../VLA_predictive_interaction_states_slides.html', import.meta.url)));
const endpoint = `http://127.0.0.1:${process.argv[3] || 9247}`;
const target = await (await fetch(`${endpoint}/json/new?about:blank`, {method: 'PUT'})).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, {once: true}); ws.addEventListener('error', reject, {once: true}); });
let sequence = 0;
const pending = new Map(), errors = [];
ws.addEventListener('message', ({data}) => {
  const message = JSON.parse(data);
  if (message.id) {
    const request = pending.get(message.id);
    pending.delete(message.id);
    message.error ? request.reject(message.error) : request.resolve(message.result);
  } else if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails);
});
const cdp = (method, params = {}) => new Promise((resolve, reject) => {
  const id = ++sequence; pending.set(id, {resolve, reject}); ws.send(JSON.stringify({id, method, params}));
});
async function evaluate(expression) {
  const result = await cdp('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
  assert(!result.exceptionDetails, JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
const viewport = (width, height) => cdp('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: false});
const key = key => evaluate(`document.body.dispatchEvent(new KeyboardEvent('keydown', {key:${JSON.stringify(key)}, bubbles:true}))`);
try {
  await cdp('Page.enable'); await cdp('Runtime.enable');
  const loaded = new Promise(resolve => ws.addEventListener('message', function listener({data}) {
    if (JSON.parse(data).method === 'Page.loadEventFired') { ws.removeEventListener('message', listener); resolve(); }
  }));
  await cdp('Page.navigate', {url: pathToFileURL(file).href}); await loaded;
  const count = await evaluate('slides.length');
  assert.equal(count, 25);
  assert.equal(await evaluate('new Set(slides.map(s => s.id)).size'), count);
  assert.equal(await evaluate('document.documentElement.lang'), 'en');
  assert(!/[\u4e00-\u9fff]/u.test(fs.readFileSync(file, 'utf8')), 'English-only content');
  assert(await evaluate('slides.every(s => s.querySelector(".notes"))'), 'Speaker notes on every slide');
  const overflow = [];
  for (const [width, height] of [[1440, 900], [1280, 720], [390, 844]]) {
    await viewport(width, height);
    for (let i = 0; i < count; i++) {
      await evaluate(`show(${i})`);
      assert.equal(await evaluate('slides.filter(s => getComputedStyle(s).display !== "none").length'), 1);
      const bounds = await evaluate('({x:slides[index].scrollWidth-slides[index].clientWidth, y:slides[index].scrollHeight-slides[index].clientHeight})');
      assert(bounds.x <= 1, `Horizontal overflow at ${width}, slide ${i + 1}`);
      if (width >= 1280 && bounds.y > 1) overflow.push({width, slide: i + 1, pixels: bounds.y});
    }
  }
  await evaluate('show(0)'); await key('ArrowRight'); assert.equal(await evaluate('index'), 1);
  await key('n'); assert(await evaluate('slides[index].querySelector(".notes").classList.contains("show")'));
  await key('Escape'); assert.equal(await evaluate('notesButton.getAttribute("aria-pressed")'), 'false');
  await evaluate('document.querySelector("#next").click()'); assert.equal(await evaluate('index'), 2);
  await evaluate('picker.value="4"; picker.dispatchEvent(new Event("change"))'); assert.equal(await evaluate('index'), 4);
  await evaluate('picker.focus(); picker.dispatchEvent(new KeyboardEvent("keydown", {key:"ArrowRight", bubbles:true}))');
  assert.equal(await evaluate('index'), 4, 'Select keyboard input must not turn pages');
  await evaluate('picker.blur(); location.hash="#othello"');
  assert.equal(await evaluate('hashIndex()'), 3);
  await evaluate('index=hashIndex(); render(); toggleOverview()');
  assert.equal(await evaluate('slides.filter(s => getComputedStyle(s).display !== "none").length'), count);
  assert(await evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile overview width');
  await key('Escape'); await key('End'); assert.equal(await evaluate('index'), count - 1);
  await key('ArrowRight'); assert.equal(await evaluate('index'), count - 1);
  await key('Home'); await key('ArrowLeft'); assert.equal(await evaluate('index'), 0);
  await evaluate('location.hash="#Infinity"'); assert.equal(await evaluate('hashIndex()'), 0);
  await evaluate('location.hash="#999"'); assert.equal(await evaluate('hashIndex()'), count - 1);
  await viewport(1280, 720); await cdp('Emulation.setEmulatedMedia', {media: 'print'});
  assert.equal(await evaluate('slides.filter(s => getComputedStyle(s).display !== "none").length'), count);
  const printOverflow = await evaluate('slides.map((s,i)=>({slide:i+1,pixels:s.scrollHeight-s.clientHeight})).filter(x=>x.pixels>1)');
  assert.deepEqual(printOverflow, [], 'Printed slide clipping');
  await cdp('Emulation.setEmulatedMedia', {media: ''});
  assert.deepEqual(errors, [], 'Browser runtime errors');
  assert.deepEqual(overflow, [], 'Desktop slides must fit without scrolling');
  if (process.argv[4]) {
    await evaluate('document.querySelector(".progress").style.transition="none"');
    const output = path.resolve(process.argv[4]);
    fs.mkdirSync(output, {recursive:true});
    for (const [width, height, index, name] of [[1440,900,0,'title'], [1440,900,3,'othello'], [1440,900,16,'comparison'], [390,844,1,'mobile']]) {
      await viewport(width, height); await evaluate(`show(${index})`);
      const shot = await cdp('Page.captureScreenshot', {format:'png'});
      fs.writeFileSync(path.join(output, `${name}.png`), Buffer.from(shot.data, 'base64'));
    }
    await viewport(1280,720);
    const pdf = await cdp('Page.printToPDF', {printBackground:true, preferCSSPageSize:true});
    fs.writeFileSync(path.join(output, 'print-preview.pdf'), Buffer.from(pdf.data, 'base64'));
  }
  console.log(JSON.stringify({slides:count, viewports:[1440,1280,390], navigation:'passed', notes:'passed', overview:'passed', print:'passed', browserErrors:errors}, null, 2));
} finally {
  await fetch(`${endpoint}/json/close/${target.id}`);
  ws.close();
}
