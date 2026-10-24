/** The first-run card and the update check show decimal megabytes, the same figure as the Terminal and the release page. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const out = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/format.ts', import.meta.url))], bundle: true, format: 'esm', platform: 'node', write: false, logLevel: 'silent' });
const { megabytes } = await import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);

test('engine sizes are decimal megabytes: 606,513,890 bytes is 607 MB, not 578', () => {
  assert.equal(megabytes(606513890), 607);
  assert.equal(megabytes(581299508), 581);
  assert.equal(megabytes(1), 1);
  assert.equal(megabytes(0), 1);
});
