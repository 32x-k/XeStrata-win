// Browser tests of the served web page (serve/web) in headless Chrome or Edge, against the mock server in mock.mjs.
// No model, no GPU. Optional: the normal test run (serve/test_web_*.py, node serve/test_web_*.js) does not need a browser.
//
//   WEBTEST_MODULES=<folder with node_modules/puppeteer-core> [CHROME_BIN=<chrome>] node tools/webtest/browser_test.mjs [name]
import assert from "node:assert/strict";
import { startMock } from "./mock.mjs";
import { launch, openPage, say, idle, sleep, runner } from "./lib.mjs";

const { test, run } = runner("web page browser tests");

const turn = (role, text, extra = {}) => ({ role, text, time: 1, ...extra });
const fence = (lang, code) => "```" + lang + "\n" + code + "\n```\n";

// ------------------------------------------------------------------------------------------------ preview (#1628)
const hostilePage = (base) => `<!doctype html><html><head><meta http-equiv="refresh" content="0;url=${base}/exfil-refresh"></head><body>
<pre id="out">static</pre><a id="l" href="${base}/exfil-link">link</a>
<form id="f" action="${base}/exfil-form" method="post"><button id="s">go</button></form>
<script>
const out = [];
const log = (k, v) => { out.push(k + ": " + v); document.getElementById("out").textContent = out.join("\\n"); };
try { log("parent.document", typeof parent.document.title); } catch (e) { log("parent.document", "blocked " + e.name); }
try { log("top.document", typeof top.document.title); } catch (e) { log("top.document", "blocked " + e.name); }
try { log("localStorage", localStorage.length); } catch (e) { log("localStorage", "blocked " + e.name); }
try { log("cookie", document.cookie.length); } catch (e) { log("cookie", "blocked " + e.name); }
try { top.location = "${base}/exfil-top"; log("top.location=", "no error"); } catch (e) { log("top.location=", "blocked " + e.name); }
try { const w = window.open("${base}/exfil-open"); log("window.open", w ? "opened" : "blocked"); } catch (e) { log("window.open", "blocked"); }
try { parent.postMessage("hello", "*"); log("postMessage", "sent"); } catch (e) { log("postMessage", "blocked"); }
fetch("${base}/exfil-fetch").then(() => log("fetch", "went out"), () => log("fetch", "blocked"));
const x = new XMLHttpRequest(); x.open("GET", "${base}/exfil-xhr"); x.onerror = () => log("xhr", "blocked"); x.onload = () => log("xhr", "went out"); try { x.send(); } catch (e) { log("xhr", "blocked"); }
new Image().src = "${base}/exfil-img";
const s = document.createElement("script"); s.src = "${base}/exfil-script.js"; document.head.appendChild(s);
try { new WebSocket("ws://127.0.0.1:9/exfil-ws"); } catch (e) { log("websocket", "blocked"); }
log("ran", "yes");
</script></body></html>`;

async function hostileChat(mock) {
  return { chat: [turn("user", "page please"), turn("assistant", "Here:\n\n" + fence("html", hostilePage(mock.url)))] };
}
const exfil = (mock) => mock.state.calls.filter((c) => c.includes("exfil"));

test("preview: an svg block previews as a picture, with no script and no network", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: { chat: [turn("user", "draw"), turn("assistant", "x\n\n" + fence("svg",
    '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80"><script>parent.__pwned=1</script><image href="' + mock.url + '/exfil-svg-img"/><circle cx="40" cy="40" r="30" fill="red" onload="parent.__pwned=2"/></svg>'))] } });
  await page.click("[data-preview-toggle]");
  const frame = await page.$("iframe");
  assert.equal(await frame.evaluate((f) => f.getAttribute("sandbox")), "", "the sandbox has no flags at all");
  assert.equal(await frame.evaluate((f) => f.getAttribute("referrerpolicy")), "no-referrer");
  const doc = await frame.evaluate((f) => f.srcdoc);
  assert.match(doc, /Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"/);
  assert.match(doc, /<img [^>]*src="data:image\/svg\+xml/, "a well-formed svg is shown as an image: images run no script");
  await sleep(400);
  assert.equal(await page.evaluate(() => window.__pwned), undefined);
  assert.deepEqual(exfil(mock), []);
  await page.close();
});

test("preview: a page does not run scripts, load addresses or leave the sandbox until asked", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: await hostileChat(mock) });
  const title = await page.title(), url = page.url();
  await page.click("[data-preview-toggle]");
  await sleep(600);
  const frame = await (await page.$("iframe")).contentFrame();
  assert.equal(await frame.evaluate(() => document.getElementById("out").textContent), "static", "the script did not run");
  assert.equal(await (await page.$("iframe")).evaluate((f) => f.getAttribute("sandbox")), "");
  assert.equal(page.lastDialog, undefined, "no dialog without the run button");
  await frame.click("#l").catch(() => {});                // a link in the page
  await frame.click("#s").catch(() => {});                // a form in the page
  await sleep(500);
  assert.deepEqual(exfil(mock), [], "no request left the preview");
  assert.equal(page.url(), url); assert.equal(await page.title(), title);
  await page.close();
});

test("preview: run with scripts keeps the page and the network out of reach", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: await hostileChat(mock) });
  const url = page.url(), title = await page.title();
  const messages = [];
  await page.evaluate(() => { window.addEventListener("message", (e) => { window.__message = e.data; }); });
  await page.click("[data-preview-toggle]");
  await page.click("[data-preview-scripts]");
  assert.match(page.lastDialog, /Run this code with scripts/, "a warning comes first");
  const iframe = await page.$("iframe");
  assert.equal(await iframe.evaluate((f) => f.getAttribute("sandbox")), "allow-scripts", "scripts, and nothing else");
  await sleep(1200);
  const frame = await iframe.contentFrame();
  const out = await frame.evaluate(() => document.getElementById("out").textContent);
  for (const line of ["ran: yes", "parent.document: blocked SecurityError", "top.document: blocked SecurityError",
                      "localStorage: blocked SecurityError", "cookie: blocked SecurityError", "fetch: blocked", "xhr: blocked", "window.open: blocked"]) {
    assert.ok(out.includes(line), `${line}\n--- got:\n${out}`);
  }
  await frame.click("#l").catch(() => {});
  await frame.click("#s").catch(() => {});
  await sleep(800);
  assert.deepEqual(exfil(mock), [], "no address was reached from the preview: " + exfil(mock));
  assert.equal(page.url(), url, "the page did not navigate");
  assert.equal(await page.title(), title);
  assert.equal(await page.evaluate(() => document.querySelectorAll("iframe").length), 1, "no frame was added to the page");
  assert.equal(await page.evaluate(() => typeof localStorage.getItem("strata.chat")), "string", "the page's own store is untouched");
  // a message to the page is the only thing that can cross; the page has no listener of its own, so nothing happens with it
  assert.ok(!(await page.evaluate(() => /addEventListener\(["']message["']/.test(document.scripts[document.scripts.length - 1].textContent || ""))));
  // closing the preview ends the script
  await page.click("[data-preview-toggle]");
  assert.equal(await page.$eval(".st-code", (b) => b.classList.contains("st-code--run")), false);
  assert.equal(await page.$eval("iframe", (f) => f.getAttribute("sandbox")), "");
  assert.equal(await page.$eval("iframe", (f) => f.getAttribute("srcdoc")), null);
  await page.close();
});

test("preview: the page's CSP header is what keeps a link inside the preview from loading a web page", async ({ browser }) => {
  const link = (m) => ({ chat: [turn("user", "x"), turn("assistant", fence("html", `<a id="l" href="${m.url}/exfil-link">go</a>`))] });
  for (const csp of [false, true]) {
    const m = await startMock({ csp });
    const page = await openPage(browser, m, { seed: link(m) });
    await page.click("[data-preview-toggle]");
    await sleep(300);
    await (await (await page.$("iframe")).contentFrame()).click("#l");
    await sleep(700);
    const went = exfil(m).length;
    assert.equal(went, csp ? 0 : 1, csp ? "with the header the link goes nowhere" : "without it the sandbox alone does let a link load a page (why the header exists)");
    await page.close();
    await m.stop();
  }
});

test("preview: declining the warning runs nothing", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: await hostileChat(mock), dialogs: "dismiss" });
  await page.click("[data-preview-toggle]");
  await page.click("[data-preview-scripts]");
  assert.ok(page.lastDialog);
  assert.equal(await page.$eval("iframe", (f) => f.getAttribute("sandbox")), "");
  assert.equal(await (await (await page.$("iframe")).contentFrame()).evaluate(() => document.getElementById("out").textContent), "static");
  await page.close();
});

test("preview: not offered for a half-written block, other languages or text that is not markup", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { seed: { chat: [turn("user", "x"),
    turn("assistant", fence("python", "<svg></svg>") + fence("html", "just words") + fence("svg", "not an svg") + "```html\n<p>unfinished")] } });
  assert.equal(await page.$$eval("[data-preview-toggle]", (e) => e.length), 0);
  await page.close();
});

test("preview: follows the theme and shows the code again", async ({ browser, mock }) => {
  const page = await openPage(browser, mock, { dark: false, seed: { chat: [turn("user", "x"), turn("assistant", fence("html", "<p>hello</p>"))] } });
  await page.click("[data-preview-toggle]");
  const doc = () => page.$eval("iframe", (f) => f.srcdoc);
  assert.match(await doc(), /color-scheme:light/);
  await page.click("#theme-btn");
  assert.match(await doc(), /color-scheme:dark/, "the frame is built again when the theme changes");
  assert.equal(await page.$eval("pre", (p) => getComputedStyle(p).display), "none");
  await page.click("[data-preview-toggle]");
  assert.notEqual(await page.$eval("pre", (p) => getComputedStyle(p).display), "none");
  await page.close();
});

test("preview: from a streamed answer, only once the block is complete", async ({ browser, mock }) => {
  const page = await openPage(browser, mock);
  await say(page, "#svg draw a circle");
  await idle(page);
  assert.equal(await page.$$eval("[data-preview-toggle]", (e) => e.length), 1);
  await page.click("[data-preview-toggle]");
  const w = await (await page.$("iframe")).evaluate((f) => f.getBoundingClientRect().width);
  assert.ok(w > 100, "the frame has room");
  await page.close();
});

// ------------------------------------------------------------------------------------------------ math (#2972, #2974)
test("math: off by default, drawn when switched on, source kept", async ({ browser, mock }) => {
  const page = await openPage(browser, mock);
  await say(page, "#math please");
  await idle(page);
  assert.equal(await page.$$eval(".katex", (e) => e.length), 0, "off: the model's text as written");
  await page.evaluate(() => { settings.math = true; renderChat(); });
  await page.waitForSelector(".katex");
  assert.equal(await page.$$eval(".katex", (e) => e.length), 3, "the inline, the display and the blocked-command formula");
  const requestText = await page.evaluate(() => JSON.stringify(apiMessages()));
  assert.ok(requestText.includes("\\\\pi r^2") && requestText.includes("\\\\frac{n(n+1)}{2}"), "the request keeps the source");
  assert.equal(await page.$$eval("#chat a[href], #chat .chat-math img", (e) => e.length), 0, "no link or picture from a formula");
  assert.deepEqual(exfil(mock), []);
  assert.equal(await page.evaluate(() => messages.at(-1).text.includes("$\\pi r^2$")), true);
  await page.close();
});

// ------------------------------------------------------------------------------------------------ load / unload (#3550)
const btnText = (page) => page.$eval("#unload-btn", (b) => b.textContent);
const waitBtn = (page, text) => page.waitForFunction((t) => document.getElementById("unload-btn").textContent === t, { timeout: 8000 }, text);
const posts = (mock, what) => mock.state.calls.filter((c) => c === "POST " + what).length;

test("load/unload: the header button unloads after a confirmation and loads again", async ({ browser, mock }) => {
  mock.state.loaded = true; mock.state.calls.length = 0;
  const page = await openPage(browser, mock);
  await waitBtn(page, "Unload");
  await page.click("#unload-btn");
  assert.match(page.lastDialog, /Unload the model and free the GPU/);
  await waitBtn(page, "Load");
  assert.equal(posts(mock, "/unload"), 1);
  assert.equal(await page.$eval("#pill-text", (e) => e.textContent), "Unloaded");
  await sleep(2500);                                       // the poll must agree, not flip the button back
  assert.equal(await btnText(page), "Load");
  await page.click("#unload-btn");
  await waitBtn(page, "Unload");
  assert.equal(posts(mock, "/load"), 1);
  assert.equal(await page.$eval("#pill-text", (e) => e.textContent), "Idle");
  await page.close();
});

test("load/unload: declining the question changes nothing", async ({ browser, mock }) => {
  mock.state.loaded = true; mock.state.calls.length = 0;
  const page = await openPage(browser, mock, { dialogs: "dismiss" });
  await waitBtn(page, "Unload");
  await page.click("#unload-btn");
  await sleep(500);
  assert.equal(posts(mock, "/unload"), 0);
  assert.equal(await btnText(page), "Unload");
  await page.close();
});

test("load/unload: not while an answer is being written", async ({ browser, mock }) => {
  mock.state.loaded = true;
  const page = await openPage(browser, mock);
  await waitBtn(page, "Unload");
  await say(page, "#hold please");
  await page.waitForFunction(() => busy !== null);
  assert.equal(await page.$eval("#unload-btn", (b) => b.disabled), true);
  assert.match(await page.$eval("#unload-btn", (b) => b.title), /Stop the reply/);
  await page.click("#stop-btn");
  await idle(page);
  await page.waitForFunction(() => !document.getElementById("unload-btn").disabled, { timeout: 8000 });
  await page.close();
});

test("load/unload: an unloaded model is loaded by a message and the button follows", async ({ browser, mock }) => {
  mock.state.loaded = false;
  const page = await openPage(browser, mock);
  await waitBtn(page, "Load");
  await say(page, "hello");
  await idle(page);
  await waitBtn(page, "Unload");
  await page.close();
});

test("load/unload: the header fits a phone", async ({ browser, mock }) => {
  mock.state.loaded = true;
  for (const width of [320, 360, 390, 640]) {
    const page = await openPage(browser, mock, { width, height: 700 });
    await waitBtn(page, "Unload");
    const r = await page.evaluate(() => {
      const box = (id) => { const b = document.getElementById(id).getBoundingClientRect(); return [b.left, b.right]; };
      return { unload: box("unload-btn"), theme: box("theme-btn"), scroll: document.documentElement.scrollWidth, width: innerWidth };
    });
    for (const [l, rt] of [r.unload, r.theme]) assert.ok(l >= 0 && rt <= r.width, `${width}px: ${JSON.stringify(r)}`);
    assert.ok(r.scroll <= r.width, `${width}px: the page scrolls sideways: ${JSON.stringify(r)}`);
    await page.close();
  }
});

const mock = await startMock();
const browser = await launch();
await run(async () => ({ browser, mock, close: async () => { await browser.close(); await mock.stop(); } }));
