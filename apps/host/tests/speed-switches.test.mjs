// The three 0.2.7 speed features are on by default and each has an off switch in Expert settings (the model's "Native engine settings").
// Off writes exactly the engine environment of the switch contract (phaseA-tests/notes-027-plan.md) into the model's settings file; on removes it,
// so a default launch carries none of it. strata_control/speed_switches.py says the same for the Terminal (tests/terminal/test_speed_switches.py).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SPEED_SWITCHES, speedSwitchApplies, speedSwitchIsOff, speedSwitchRows, setSpeedSwitch, validateNativeSettings, publicNativeEnv, restoreNativeEnv } from '../server/native-settings.mjs';
import { nativeFixture } from './helpers/native-fixture.mjs';

const read = rel => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8').replace(/\r\n/g, '\n');
const OFF = {
  faster_long_prompts: { STRATA_KVA: '0' },
  faster_radeon_9000: { STRATA_PCIE_FRAC_DEFAULT: 'old' },
  deepseek_speedups: { DS4_FAST_ATTN: '0', DS4_ATTN_SLAB: '0', DS4_COPY_PIN: '0', DS4_COPY_HUGETLB: '0', DS4_STORE_SPARE: '0', DS4_STORE_PUT: '0' },
};
const R9700 = { index: 0, name: 'AMD Radeon AI PRO R9700', vendor: 'amd', arch: 'gfx1201', vram_gb: 32 };
const RX7900 = { index: 0, name: 'AMD Radeon RX 7900 XTX', vendor: 'amd', arch: 'gfx1100', vram_gb: 24 };
const RTX = { index: 0, name: 'NVIDIA GeForce RTX 4090', vendor: 'nvidia', arch: '89', vram_gb: 24 };
const STRIX = { index: 0, name: 'AMD Radeon 8060S', vendor: 'amd', arch: 'gfx1151', vram_gb: 96, uma: true };

test('there are exactly three switches and each writes the engine names of the contract', () => {
  assert.deepEqual(SPEED_SWITCHES.map(s => s.id), Object.keys(OFF));
  for (const sw of SPEED_SWITCHES) assert.deepEqual(sw.off, OFF[sw.id], sw.id);
});

test('a default launch has none of the new environment entries: every switch reads as on and an empty environment stays empty', () => {
  for (const sw of SPEED_SWITCHES) assert.equal(speedSwitchIsOff({}, sw.id), false);
  const env = { PATH: '/native/bin', STRATA_PREFILL: '2' };
  for (const sw of SPEED_SWITCHES) assert.deepEqual(setSpeedSwitch(env, sw.id, true), env, 'on removes nothing that is not the switch\'s');
  const rows = speedSwitchRows({}, { backend: 'hip', gpus: [R9700] });
  assert.ok(rows.every(r => r.on === true));
});

test('each switch off yields exactly its entries and nothing else; on again gives back the same environment', () => {
  const env = { PATH: '/native/bin', STRATA_PREFILL: '2', DS4_ASYNC_T: '4' };
  for (const sw of SPEED_SWITCHES) {
    const off = setSpeedSwitch(env, sw.id, false);
    assert.deepEqual(off, { ...env, ...OFF[sw.id] }, sw.id);
    assert.equal(speedSwitchIsOff(off, sw.id), true);
    for (const other of SPEED_SWITCHES) if (other.id !== sw.id) assert.equal(speedSwitchIsOff(off, other.id), false, `${sw.id} does not turn ${other.id} off`);
    const back = setSpeedSwitch(off, sw.id, true);
    assert.deepEqual(back, env, sw.id);
    assert.equal(JSON.stringify(back), JSON.stringify(env), 'byte for byte, key order included');
  }
  assert.throws(() => setSpeedSwitch({}, 'nothing', false), /Unknown speed switch/);
});

test('a hand-edited file with only some of a switch\'s entries reads as on and is left alone by an unchanged save', () => {
  const partial = { DS4_FAST_ATTN: '0' };
  assert.equal(speedSwitchIsOff(partial, 'deepseek_speedups'), false);
  assert.deepEqual(setSpeedSwitch(partial, 'faster_long_prompts', true), partial);
  assert.equal(speedSwitchIsOff({ STRATA_KVA: '1' }, 'faster_long_prompts'), false, 'only the off value counts');
});

test('a switch shows only where it applies: Qwen on CUDA or HIP, Qwen on a Radeon RX 9000 / R9700, DeepSeek', () => {
  const ids = context => speedSwitchRows({}, context).map(r => r.id);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [RX7900] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [STRIX] }), ['faster_long_prompts'], 'Strix Halo (gfx1151) is not gfx12');
  assert.deepEqual(ids({ backend: 'hip', gpus: [R9700] }), ['faster_long_prompts', 'faster_radeon_9000']);
  assert.deepEqual(ids({ backend: 'hip', gpus: [{ name: 'AMD Radeon RX 9070 XT', vendor: 'amd' }] }), ['faster_long_prompts', 'faster_radeon_9000'], 'no arch reported: the card name decides');
  assert.deepEqual(ids({ backend: 'hip', gpus: [{ name: 'AMD Radeon RX 7900 XT', vendor: 'amd' }] }), ['faster_long_prompts']);
  assert.deepEqual(ids({ backend: 'cuda', gpus: [RTX, R9700] }), ['faster_long_prompts'], 'the Radeon switch needs the HIP engine');
  for (const backend of ['sycl', 'metal', 'cpu', null]) assert.deepEqual(ids({ backend, gpus: [R9700] }), [], `Qwen on ${backend}`);
  assert.deepEqual(ids({ deepseek: true, backend: 'hip', gpus: [R9700] }), ['deepseek_speedups'], 'DeepSeek has its own switch only');
  assert.deepEqual(ids({ deepseek: true, backend: 'cuda', gpus: [RTX] }), ['deepseek_speedups']);
  assert.equal(speedSwitchApplies('deepseek_speedups', { deepseek: false, backend: 'cuda', gpus: [RTX] }), false);
  assert.deepEqual(ids({}), [], 'nothing is known: nothing is offered');
});

test('the environment names round-trip through the validator, the public view and the restore, and other names stay untouched', () => {
  const all = Object.assign({}, ...Object.values(OFF));
  assert.deepEqual(validateNativeSettings({ env: all }), { env: all });
  assert.throws(() => validateNativeSettings({ env: { HOME: '/tmp' } }));
  assert.throws(() => validateNativeSettings({ env: { DS4_fast: '0' } }));
  const previous = { PATH: '/native/bin', HIP_VISIBLE_DEVICES: '0', ...all };
  assert.deepEqual(publicNativeEnv(previous), all);
  assert.deepEqual(restoreNativeEnv({}, previous), { PATH: '/native/bin', HIP_VISIBLE_DEVICES: '0' }, 'turning every switch on removes only theirs');
  assert.deepEqual(restoreNativeEnv(all, { PATH: '/native/bin' }), { PATH: '/native/bin', ...all });
});

function setConfig(f, model, patch) {
  const path = join(f.repo, `strata-${model}.json`);
  writeFileSync(path, JSON.stringify({ ...JSON.parse(readFileSync(path, 'utf8')), ...patch }));
  return path;
}

test('through the host: the settings answer lists the switches that apply, and off then on gives back the same file', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 64, gpus: [R9700] });
    const path = setConfig(f, 'iq2_xs', { backend: 'hip' });
    await f.control.refresh();
    const before = readFileSync(path, 'utf8');
    const first = f.control.settings('iq2_xs');
    assert.deepEqual(first.speedSwitches.map(s => [s.id, s.on]), [['faster_long_prompts', true], ['faster_radeon_9000', true]]);
    assert.deepEqual(first.speedSwitches.map(s => s.off), [OFF.faster_long_prompts, OFF.faster_radeon_9000], 'the client writes what the host says');
    assert.equal('STRATA_KVA' in first.values.env, false, 'default: nothing in the environment');

    // the client turns "Faster long prompts" off: it sends the environment with the entry added
    const off = f.control.save('iq2_xs', { values: { env: { ...first.values.env, ...OFF.faster_long_prompts } } });
    const file = JSON.parse(readFileSync(path, 'utf8'));
    assert.equal(file.env.STRATA_KVA, '0');
    assert.equal(file.env.PATH, '/native/bin');
    assert.equal(file.env.STRATA_TOKEN, 'private', 'a credential is kept, not lost behind its mask');
    assert.equal('STRATA_PCIE_FRAC_DEFAULT' in file.env, false);
    assert.deepEqual(off.speedSwitches.map(s => [s.id, s.on]), [['faster_long_prompts', false], ['faster_radeon_9000', true]]);

    // and on again: the entry is gone and the file is as it was
    const env = { ...off.values.env };
    delete env.STRATA_KVA;
    const on = f.control.save('iq2_xs', { values: { env } });
    assert.deepEqual(on.speedSwitches.map(s => s.on), [true, true]);
    const after = JSON.parse(readFileSync(path, 'utf8'));
    assert.deepEqual(after, JSON.parse(before));
    assert.equal(JSON.stringify(after.env), JSON.stringify(JSON.parse(before).env), 'the same environment, same order');
  } finally { await f.cleanup(); }
});

test('through the host: a DeepSeek model gets its own switch only, and the Strix async entry is kept beside it', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 128, gpus: [STRIX] });
    const path = setConfig(f, 'iq2_xs', { arch: 'deepseek41', backend: 'hip', env: { DS4_ASYNC_T: '4' } });
    await f.control.refresh();
    const first = f.control.settings('iq2_xs');
    assert.deepEqual(first.speedSwitches.map(s => [s.id, s.on]), [['deepseek_speedups', true]]);
    assert.equal(first.values.env.DS4_ASYNC_T, '4', 'setup\'s own entry is visible and kept');
    const off = f.control.save('iq2_xs', { values: { env: { ...first.values.env, ...OFF.deepseek_speedups } } });
    assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')).env, { DS4_ASYNC_T: '4', ...OFF.deepseek_speedups });
    assert.equal(off.speedSwitches[0].on, false);
    const env = Object.fromEntries(Object.entries(off.values.env).filter(([name]) => !(name in OFF.deepseek_speedups)));
    f.control.save('iq2_xs', { values: { env } });
    assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')).env, { DS4_ASYNC_T: '4' });
  } finally { await f.cleanup(); }
});

test('through the host: a Qwen model on a PC the engine cannot speed up offers no switch (SYCL, a Mac, no card)', async () => {
  const f = nativeFixture();
  try {
    f.setHardware({ ram_gb: 32, gpus: [{ index: 0, name: 'Intel Arc A750', vendor: 'intel', vram_gb: 8 }] });
    setConfig(f, 'iq2_xs', { backend: 'sycl' });
    await f.control.refresh();
    assert.deepEqual(f.control.settings('iq2_xs').speedSwitches, []);
  } finally { await f.cleanup(); }
});

test('the Expert settings show the rows in the existing row layout with an On / Off choice, and the words are in all eight languages', () => {
  const screen = read('../src/components/NativeSettings.tsx');
  assert.match(screen, /<h3 className="h4">\{tr\("Speed"\)\}<\/h3>/);
  assert.match(screen, /<label className="native-setting-row" key=\{sw\.id\}><span><strong>\{tr\(sw\.label\)\}<\/strong><small>\{tr\(sw\.hint\)\}<\/small><\/span><Select /, 'the same row, the same Select');
  assert.match(screen, /label: 'On \(recommended\)' \}, \{ value: 'off', label: 'Off' \}/);
  assert.match(screen, /if \(on\) delete env\[name\]; else env\[name\] = value;/, 'on removes the entries');
  const languages = ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko'];
  const catalog = lang => JSON.parse(read(`../src/locales/fix-${lang}.json`));
  const words = [...SPEED_SWITCHES.flatMap(s => [s.label, s.hint]), 'On (recommended)'];
  const english = Object.values(catalog('en'));
  for (const word of words) assert.ok(english.includes(word), `English catalog: ${word}`);
  for (const lang of languages.slice(1)) {
    const translated = Object.values(catalog(lang));
    const en = catalog('en');
    for (const [id, text] of Object.entries(en)) if (words.includes(text)) {
      assert.ok(catalog(lang)[id] && catalog(lang)[id] !== text, `${lang} ${id} is translated`);
      assert.ok(translated.includes(catalog(lang)[id]));
      assert.ok(!/\{\d+\}/.test(catalog(lang)[id]));
    }
  }
  for (const lang of ['zh', 'ja', 'ko']) assert.ok(Object.values(catalog(lang)).some(v => v.includes('0.2.6')), 'the version number is kept in the DeepSeek text');
});

test('the effect texts say what turning the feature off does, in plain words', () => {
  for (const sw of SPEED_SWITCHES) {
    assert.match(sw.hint, /[Tt]urn off/, sw.id);
    assert.ok(sw.hint.length < 260, sw.id);
    assert.ok(!/STRATA_|DS4_|env|--/.test(sw.hint), 'no flags or variable names for a non-technical reader');
  }
  assert.equal(SPEED_SWITCHES[0].hint, 'Reads long prompts faster by approximating part of the work. Turn off if answers about long documents look wrong.');
});
