// Context pressure and compaction (2026-10-22 hands-on test): compaction reads the size the model server measured, a summary request
// always fits the window, and a refused prompt is recovered from. The Harness is real; the model server is the scripted worker, set to
// count a prompt the way the engine does (characters over a density) and to refuse one that leaves no room.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startScriptedWorker } from './helpers/scripted-worker.mjs';
import { createAgentBridge } from '../server/agent/agent-bridge.mjs';
import { resolveHarness } from '../server/agent/harness-runtime.mjs';
import { buildProfile } from '../server/agent/profile.mjs';
import { clipMessage, densityOf, estimateMessage, isContextOverflow, pressureNow, promptTokensOf, splitMessages, summaryBudget, usageOfEvent } from '../server/agent/context-budget.mjs';

const HARNESS = process.env.STRATA_HARNESS_DIR;
const skip = !HARNESS || !resolveHarness({ env: process.env }).ok ? 'STRATA_HARNESS_DIR is not set to a built Harness' : false;

test('the profile loads the context plugin, so compaction and /compact read the measured prompt', () => {
  const profile = buildProfile({ local: [{ id: 'a', modelId: 'm', baseUrl: 'http://x', context: 65536 }], connections: [] });
  assert.match(profile.patch, /"id": "strata-context"/);
  assert.match(profile.patch, /strata-context\.mjs/);
});

test('density: the engine said 59,048 tokens for a prompt the Harness priced at 31,794', () => {
  const d = densityOf(59048, 31794);
  assert.ok(d > 1.85 && d < 1.87, `density ${d}`);
  assert.equal(densityOf(0, 100), 1);
  assert.equal(densityOf(100, 0), 1);
  assert.equal(densityOf(10, 100), 1, 'plain prose is never priced lower than the Harness already does');
  assert.equal(densityOf(10000, 100), 6, 'a wild sample is capped');
});

test('pressure: the measured part is exact, what came after is priced at the measured density, and it never reads lower than the Harness', () => {
  const sample = { real: 59048, estimated: 31794 };
  assert.equal(pressureNow({ sample, now: 31794 }), 59048, 'nothing added since the sample: the measured number');
  const grown = pressureNow({ sample, now: 31794 + 1000 });
  assert.ok(grown > 59048 + 1800 && grown < 59048 + 1900, `1000 added tokens are worth about 1,860 real ones (${grown})`);
  assert.ok(pressureNow({ sample, now: 31794 - 5000 }) < 59048, 'a compaction lowers it');
  assert.equal(pressureNow({ sample: null, now: 1234 }), 1234, 'no measurement yet: the Harness price');
  assert.equal(pressureNow({ sample: { real: 100, estimated: 5000 }, now: 5000 }), 5000, 'never below the Harness price');
});

test('the compaction threshold on a 64K window leaves room for the reply even when the prompt is measured in real tokens', () => {
  // Gyro-S at 64K: the reply keeps 19,660 tokens and compaction keeps 5,734 of headroom (profile.mjs), so it starts at 40,142 REAL tokens.
  const window = 65536, reply = 19660, headroom = 5734;
  const threshold = Math.floor(Math.min(window * 0.8, window - reply - headroom));
  assert.equal(threshold, 40142);
  assert.ok(threshold + reply < window, 'the prompt at the threshold plus a full reply still fits');
  // The 2026-10-22 chat: 31,794 priced = 59,048 real. It is over the threshold now (before: under it, and the engine refused the next request).
  assert.ok(pressureNow({ sample: { real: 59048, estimated: 31794 }, now: 31794 }) > threshold);
});

test('usage helpers: prompt side only, from a message or from its stream', () => {
  assert.equal(promptTokensOf({ inputTokens: 100, outputTokens: 50, cacheReadTokens: 30, cacheWriteTokens: 5 }), 135);
  assert.equal(promptTokensOf(null), 0);
  assert.deepEqual(usageOfEvent({ data: { usage: { inputTokens: 7 } } }), { inputTokens: 7 });
  assert.deepEqual(usageOfEvent({ data: { stream: [{ type: 'chunk', chunk: { type: 'text-delta' } }, { type: 'chunk', chunk: { type: 'usage', usage: { inputTokens: 9 } } }] } }), { inputTokens: 9 });
  assert.equal(usageOfEvent({ data: {} }), null);
});

test('a refusal for the context window is recognised by its code and by the engine\'s own words', () => {
  assert.ok(isContextOverflow({ code: 'CONTEXT_WINDOW_EXCEEDED', message: 'x' }));
  assert.ok(isContextOverflow(new Error('400: {"type":"invalid_request_error","message":"request exceeds the context window: prompt (70141 tokens) leaves no room to answer in the context (65536); requests are never truncated"}')));
  assert.ok(!isContextOverflow(new Error('fetch failed')));
  assert.ok(!isContextOverflow(null));
});

test('a summary request is split into pieces that fit, each message whole when it can be, a huge one cut in the middle', () => {
  const text = n => ({ role: 'user', content: [{ type: 'text', text: 'x'.repeat(n) }] });
  const many = Array.from({ length: 12 }, () => text(4000));   // 1,000 tokens each by the Harness price
  const runs = splitMessages(many, 3500, 1);
  assert.deepEqual(runs.map(r => r.length), [3, 3, 3, 3]);
  assert.deepEqual(runs.flat().length, 12, 'nothing is dropped');
  const dense = splitMessages(many, 3500, 2);
  assert.ok(dense.length > runs.length, 'a denser text means more pieces');
  const huge = clipMessage(text(400000), 8000, 1);
  assert.ok(estimateMessage(huge) <= 8000, `cut to its share (${estimateMessage(huge)})`);
  assert.match(huge.content[0].text, /left out of this summary request/);
  assert.equal(clipMessage(text(100), 8000, 1).content[0].text.length, 100, 'a small message is untouched');
  const oneBig = splitMessages([text(10), text(600000), text(10)], 6000, 1);
  assert.ok(oneBig.every(r => r.reduce((n, m) => n + estimateMessage(m), 0) <= 6000), 'every piece fits after the cut');
  // A tool call and its result stay together when a clean cut exists.
  const call = { role: 'assistant', content: [{ type: 'tool-call', id: 'c', name: 'read', arguments: '{}' }] };
  const result = { role: 'tool', toolCallId: 'c', content: [{ type: 'text', text: 'y'.repeat(2000) }] };
  const paired = splitMessages([text(2000), call, result, text(2000)], 1200, 1);
  assert.ok(paired.every(r => !r.includes(call) || r.includes(result)), 'the call and its result are in one piece');
  assert.ok(summaryBudget({ window: 65536, summaryTokens: 5734, toolsTokens: 10133, systemTokens: 400 }) > 40000);
});

/** The scripted model server counting like the engine (3 characters a token here: logs and code) in a 32K window. */
async function rig(t, { window = 32768, charsPerToken = 3 } = {}) {
  const worker = await startScriptedWorker({ window, charsPerToken });
  const dir = mkdtempSync(join(tmpdir(), 'strata-ctx-'));
  const ws = join(dir, 'ws'); mkdirSync(ws, { recursive: true });
  const bridge = createAgentBridge({ dataDir: dir, localInstances: () => [{ id: 'i', label: 'S', baseUrl: worker.url, modelId: 'scripted-model', context: window, slots: 2 }], connections: () => [], env: process.env, startTimeoutMs: 90_000 });
  t.after(async () => { await bridge.close(); await worker.close(); try { rmSync(dir, { recursive: true, force: true, maxRetries: 3 }); } catch { /* best effort */ } });
  const send = async (message) => { const frames = []; await bridge.runTurn({ chatId: 'c', message, workspacePath: ws, mode: 'bypass', onFrame: f => frames.push(f) }); return frames; };
  const harnessLog = () => { try { return readFileSync(join(dir, 'agent', 'harness.log'), 'utf8'); } catch { return ''; } };
  return { worker, bridge, send, ws, harnessLog };
}
const answerOf = frames => frames.filter(f => f.type === 'delta').map(f => f.text).join('');
const dense = 'q=0x1f ok;';   // 10 characters: 2.5 tokens by the Harness price, 3.3 by the model server's count here

test('the plugin loads in the real Harness (a plugin that fails to import would leave compaction as it was, silently)', { skip }, async (t) => {
  const { send, harnessLog } = await rig(t);
  const frames = await send('SCRIPT:echo hello');
  assert.equal(frames.at(-1).type, 'done');
  assert.doesNotMatch(harnessLog(), /did not activate|failed to import|strata-context/, harnessLog());
});

test('dense text compacts in time: no request is refused, and every message is answered', { skip }, async (t) => {
  const { worker, send } = await rig(t);
  const replies = [];
  for (let i = 0; i < 8; i++) {
    const frames = await send(`SCRIPT:echo round${i} ${dense.repeat(900)}`);
    assert.equal(frames.at(-1).type, 'done', `round ${i} ended normally: ${JSON.stringify(frames.at(-1)).slice(0, 300)}`);
    replies.push(answerOf(frames).slice(0, 12));
  }
  assert.ok(replies.every(r => /^round[0-9]/.test(r)), replies.join(' | '));
  const refused = worker.requests.filter(r => r.refused);
  assert.equal(refused.length, 0, `the model server refused ${refused.length} request(s): ${refused.map(r => r.refused).join(', ')} tokens`);
  assert.ok(worker.requests.some(r => r.summary), 'the older history was summarised');
  // Compaction starts before the next message would push the prompt past the threshold, so no answer was asked for from a nearly full window.
  const peak = Math.max(...worker.requests.filter(r => !r.summary).map(r => Math.round(r.promptChars / 3)));
  assert.ok(peak < 32768 - 4000, `no prompt came within 4,000 tokens of the window (peak ${peak})`);
});

test('a chat the model server refused once is not stuck: the next message compacts (the summary itself fits the window) and goes through', { skip }, async (t) => {
  // 2026-10-22: a 103 KB message made a 70K-token prompt, the 400 left it in the history, and then neither the next message nor /compact
  // could go through, because the summary request carried the same history and was refused for its size too.
  const { worker, send } = await rig(t);
  assert.equal((await send(`SCRIPT:echo one ${dense.repeat(300)}`)).at(-1).type, 'done');
  const huge = await send(`SCRIPT:echo two ${dense.repeat(7000)}`);
  assert.equal(huge.at(-1).type, 'done', 'the Harness compacted after the refusal and asked again: ' + JSON.stringify(huge.at(-1)).slice(0, 200));
  const small = await send('SCRIPT:echo three please');
  assert.equal(small.at(-1).type, 'done', JSON.stringify(small.at(-1)).slice(0, 300));
  assert.match(answerOf(small), /three/);
  const again = await send('SCRIPT:echo four please');
  assert.equal(again.at(-1).type, 'done');
  const refused = worker.requests.filter(r => r.refused);
  assert.equal(refused.length, 1, `only the first, too large, prompt was refused: ${refused.map(r => r.refused).join(', ')}`);
  assert.ok(worker.requests.filter(r => r.summary).every(r => Math.round(r.promptChars / 3) < 32768 - 64), 'every summary request fitted the window');
});
