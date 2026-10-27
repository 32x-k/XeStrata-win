# Strata 0.2.7: faster 12 GB NVIDIA cards, DeepSeek on Ryzen AI Max, a bigger web chat page, French

The app and the Terminal are 0.2.7. Control and every engine package (Qwen and DeepSeek) are 0.1.52, because both engines changed.

DRAFT of 27 October. Text in square brackets that starts with "pending" or "TO CONFIRM" is work that was not merged or checked when this was written. `phaseA-tests\rel027\check_notes027.py` lists them and removes the ones that landed. None of them may be left in the published text.

## What you will notice

- **NVIDIA cards with 12 GB or less answer faster.** The router now prefers experts that are already on the card, which saves page-ins from RAM. On an RTX 3060 12 GB (Qwen3.8-Flash-Next IQ3_XXS, five paired runs against the same build with the setting off), a story answer went from 46.2 to 53.1 tok/s (+15%) and a code answer from 45.7 to 62.5 tok/s (+37%). It picks a slightly different set of experts, so the wording of answers changes. On 19 quick test tasks (retrieval, extraction, code, JSON, unanswerable questions, agent loops) we found no copy, code or JSON defect at this setting; a stronger setting (2.0) did show defects, so the default is 1.5. [pending: decision rp15 - the 61-task check on the same card read 49 passes against 54 with the setting off, with two real defects (an incoherent agent paragraph and a code answer that ran on in comments); the owner decides 1.5 or 1.0 from `notes-027-cval.md`, and this bullet is rewritten for the result] It applies to a discrete NVIDIA card of 13 GiB or less; larger cards, Radeon and Strix Halo are not changed. Off: "Faster answers on 12 GB NVIDIA cards" in the model's settings, or `STRATA_ROUTE_PRIOR=0`.
- **Radeon RX 9000 and AI PRO R9700: faster Qwen.** Two changes: the large per-token lookup table stays in RAM when the PC has room for it, and the decode split between card and processor starts at a better value. One R9700 (32 GB) with 125 GB of RAM on Linux, Qwen IQ3_S, medians of three runs:

  | Context | Prompt before | Prompt now | Decode before | Decode now |
  |---|---|---|---|---|
  | 8K | 2,047 tok/s | 2,505 tok/s (+22%) | 82.9 tok/s | 94.6 tok/s (+14%) |
  | 32K | 2,491 tok/s | 2,802 tok/s (+12%) | 64.9 tok/s | 85.1 tok/s (+31%) |

  Answers can differ in wording because the work is split differently; 30 test tasks passed in every arm with no invented fact. Off: "Faster answers on Radeon RX 9000 / AI PRO R9700".
- **DeepSeek-V4.1-Flash runs on Ryzen AI Max (Strix Halo) PCs with Windows.** It has its own engine package for the gfx1151 chip, and the app or `strata launch` downloads it with the model. The new guide [docs/STRIX_HALO_WINDOWS.md](STRIX_HALO_WINDOWS.md) has the numbers and the traps, measured on a GMKtec EVO-X2 (Ryzen AI Max+ 395, 128 GB shared memory, Adrenalin 26.10.1). DeepSeek 2.00 answers a short chat at 12 to 14 tok/s and reads a 4K prompt at 214 to 226 tok/s, a 16K prompt at 326 and a 32K prompt at 342 tok/s. Qwen IQ3_S on the same PC: chat 44 tok/s, 4K prompt 755, 32K prompt 792 tok/s. [TO CONFIRM: re-measure the DeepSeek and Qwen rows on the final build if strix is lent before the release] The shared-expert stream is off by default on Windows there, because it is slower: 43.6 to 46.6 tok/s decode (+6.9%, three paired starts), same token ids. Only the 128 GB machine was tested. You need Adrenalin 26.10.1 or newer.
- **DeepSeek reads prompts faster on most PCs.** The engine no longer re-reads a batch of experts from the SSD after every prompt, picks the attention buffer size by itself and fills the RAM store more cleverly. On an RTX 5070 (12 GB, 64 GB RAM, 2.25 set, three paired runs against an earlier 0.2.7 candidate): 4K prompt 116 to 170 tok/s (+47%), 8K prompt 202 to 268 tok/s (+33%). On one R9700 with the 2.25 set: 32K prompt 382 to 444 tok/s (+16%), 8K prompt 200 to 286 tok/s (+43%, but that prompt is limited by SSD reads and varies a lot between runs). The cost: the first words after a long prompt can come slower, because the store is no longer pre-filled. On the RTX 5070 a long answer still finished 8.7% sooner overall (137.5 s against 150.6 s); on one R9700 a 300-token answer after an 8K prompt took 51.9 s without the pre-fill and 48.6 s with it. `DS4_SETTLE_STORE=1` brings the pre-fill back. The speed-up switch "DeepSeek speed-ups" turns the new behaviour off in one step.
- **Four Radeon AI PRO R9700 cards run DeepSeek faster, with five experts per word by default.** With the launcher's own settings, DeepSeek 2.25: an 8K prompt 342 to 399 tok/s on the first read and 967 to 1,350 once warm (+17%, +40%), a 32K prompt 1,127 to 1,604 first and 1,183 to 1,697 warm (+42%, +43%), decode on code 88.9 to 95.0 tok/s (+7%) and on an agent task 73.5 to 79.7 (+8%). Loading takes 58 to 60 s instead of 51 s. The default is now "Faster" (5 of 6 experts per word); the quality run passed 19 of 19 tasks against 18 of 19 with six experts, and the distance to the reference model is the same (0.1102 against 0.1102). Choose Full quality in the app, or `--experts 6`, to use six.
- **DeepSeek can use 512K and 1M contexts.** Setup, the Terminal and the app offer them up to the limit of the model files, with a plain "not tested yet" note past 256K. The default stays 8K, and the app now loads a DeepSeek model with the context you chose. On an RTX 5070 a 1M context loaded in 18 s and cost about 1 GB of graphics memory, which pushed 136 experts off the card. We did not test long prompts at these sizes. The idea comes from Project Maya v1.1.28 (#442, by @needmorevram).
- **More cards get small gains from kernel changes that were measured.** On an RTX PRO 5000 (72 GB, Qwen IQ3_S, five paired runs): story answers 195.9 to 211.8 tok/s (+8.1%), code 240.6 to 244.7 (+1.7%), prompts +1.2% at 4K, +1.5% at 8K, +1.6% at 16K. On two RTX A4000: story +2.2%, code +0.5%. These switches sum numbers in another order, so a reply can differ from 0.2.6 at a near-tie. RTX 40 cards (compute capability 8.9) use measured speed tables for the quantised matrix products; they are bit for bit the same as before and we have no Ada card, so the reporter's measurements are the only ones. [pending: f-027-gh-eng] A100-class cards (sm_80) get the bitwise-identical defaults of #3779 (wider dequantisation, tuned tables); its many opt-ins stay off, and the reporters are asked to test. [pending: f-027-gh-eng]
- **The served web chat page gets a large update.** Wide windows use the width and narrow screens fit (#3176). Saved chats with search, rename and delete, editing a question, rewinding to an earlier turn and branching a new chat from it, sending a message while the answer is still coming (wait, send after, or interrupt), math written in LaTeX rendered with KaTeX, a sandboxed preview of HTML and SVG the model writes, and Load and Unload buttons in the header. The chat view is translated into the same languages as the app, French included. The history features are off until you switch them on in Chat settings: with them off the page sends the same requests as before. Monitor, Images and About are still English.
- **The desktop app.** French is the ninth language. Settings now has: a sound when a task finishes (on request), a desktop notification when Manual mode needs your approval, minimise to the system tray, a fixed port so API clients keep working across updates (with a warning when the port is busy), and an interface size up to 200%. The Monitor has a readings strip: power draw, an electricity price you set (in your local currency, found from the system region), disk temperature where Windows or Linux can report it. The DeepSeek card says how much free disk the set needs. Not tried inside the real packaged app on screen yet: tray icon, the native notification and the busy-port warning. [TO CONFIRM: run once in the release e2e]
- **Speed switches.** In Models, Native engine settings, Speed (and in the Terminal's Advanced table): "Faster long prompts" (off by default), "Faster answers on 12 GB NVIDIA cards", "Faster answers on Radeon RX 9000 / AI PRO R9700", "DeepSeek speed-ups". The Terminal also has "Qwen: 0.2.6 behaviour" for troubleshooting.

## Going back to 0.2.6

Two switches bring back every default that 0.2.7 changed, in one step, for a test or for a PC where something looks wrong.

- **`DS4_DEFAULTS=026`** (DeepSeek): the staging ring, the prompt chunk rule, the attention buffer, the pinned copy and the huge-page arena, the matrix-core attention, the drafter's cut, the five-experts default, the RAM store's spare slots and its pre-fill after a prompt, the dense dequantiser, the head and gate/up operand formats and the RDNA3 dense GEMM go back to what 0.2.6 did.
- **`STRATA_DEFAULTS=026`** (Qwen): residency-biased routing, the wide and split hyper-connection reads, the int8 prompt attention, the four gfx12 decode switches, the PLE residency policy, the gfx12 decode split (back to 0.55), the RTX 40 tables [pending: f-027-gh-eng] and the sm_80 defaults [pending: f-027-gh-eng].

A variable or flag you set yourself still wins over the switch, and the older single switches work as before. In the app, "DeepSeek speed-ups" off sends `DS4_DEFAULTS=026`.

## Off unless you turn it on

- **Faster long prompts** (`STRATA_KVA=1`, or the switch in Models, Native engine settings, Speed). Reads the last 12 layers of a very long prompt in an approximate way, for Qwen3.8-Flash-Next on one NVIDIA or AMD card, on the first read of a prompt of 4,096 tokens or more. On an RTX 5070 with IQ3_S it was 11% faster at 14K and 18% faster at 30K. It stays off because earlier, more aggressive settings dropped records from long lists and failed agent tasks. The setting that ships when you turn it on (layers 36 to 47 approximate, the last 2,048 tokens exact) found the same 585 of 606 facts as the exact read and finished 6 of 10 agent tasks where the exact read finished 4 and 7 in two runs. It can still lose details. [pending: decision kva2 - split 24 plus MoE skip is confirmed on four boxes only if the owner takes `f-027-kva2`; the numbers are in `notes-027-kva.md`]
- **Prefill on one card, decode on another** (`--split-serve`, #4000). The idea and a manual setup came from @MarcoTool. One engine reads the prompt and hands the finished state to a second engine that answers, with the state in a file. The words are the same as with one card (checked at 4,096 and 16,384 tokens: tokens and end state identical). [pending: f-027-split - TO FILL the speed table from `notes-027-split.md` before this bullet is kept; the first runs were slower than a layer split]
- **Pipelined drafts** (`STRATA_PIPED_MTP`, #4121 by @wannagoxuf): the draft model also runs for batch slots. On two RTX A6000 cards with a batch of slots, two streams +15.4% (five of five pairs), four streams +4.7%, one stream +0.1%. [pending: f-027-gh-eng-pm2]
- **Layer-major prefill on a CUDA layer split** (#3976, ports of two contributors). [pending: f-027-gh-eng-lm - state whether it is on or off and give the measured prompt gain, or delete the bullet]
- **Huge pages for DeepSeek on Linux.** If you reserve 2 MB huge pages (`vm.nr_hugepages`) big enough for the expert copy, the engine uses them and copies prompt data in place: an 8K prompt on the 2.25 set read 651 tok/s without and 694 with (R9700 limited to 12 GB; that run also set two buffers by hand). Without a reserved pool nothing changes. Not run with a reserved pool on the release package.
- **Settings that help if your PC fits them, no engine change:** `--ple-io ram` (the 28.8 GB table in RAM; on the 72 GB card prompts +16.5% at 1K, +30.6% at 4K, +16.8% at 8K), `--mtp-draft-vocab draft_vocab_en.bin` (+2.7 to +3.1% decode for English text), `--slots 2` for two conversations at once (+10% to +12% aggregate throughput on RTX PRO 5000 and RTX 3060), `--prefill-pipe` on a two-card layer split (+15% at 4K, +23% at 8K prompts on two RTX A4000). `--kv q4_0` on a 12 GB card did not reproduce a gain, so it is a choice, not a default.

## Fixed, from your reports

Each issue number gets its own reply.

- Chat: scrolling up while an answer streams works (#4014); a request that timed out shows a badge (#3991).
- The Monitor asks the NVIDIA tool only while it is open or a model runs, which stops the Windows stutter (#4066). The app no longer insists on CUDA when you have a HIP card on Windows (#3892), and the source-checkout host starts an engine with an empty API key (#3851). Deleting a model in the app removes all files of a multi-file model and leaves your own files (#3716).
- The Terminal: `strata launch --set dspark=on` works (#3959); `launch --show-key` knows the config setup wrote (#3570); a model on disk in a differently cased folder is found on Linux instead of being offered for download again (#3996); `--spec 5` can be saved with a warning (#3844) [pending: f-027-gh-eng].
- Manual mode: read-only commands that have write or execute options now ask first (#3975).
- Engine [pending: f-027-gh-eng]: a conversation cache no longer reuses another conversation's keys and values (#1503); a short draft-vocabulary row can't be written past (#3911); a checkpoint save that runs out of memory no longer exits the engine (#3350); a hand build for RTX 50 no longer ends up as sm_75 (#4029, #4023); the DeepSeek CUDA packages now include RTX 40 cards (sm_89, #3935); an undeclared tool call returned as text is recovered, declared-only kept (#3796).
- Setup and Docker: setup accepts engines that come from the update feed (#4027); a silent re-download in Docker fails loudly (#3421); the Docker image stage carries the scripts the Terminal needs (#3974, #3644); images are tagged 0.2.7.
- Docs: TROUBLESHOOTING is back with every entry that the 0.2.5 rewrite dropped, updated for 0.2.7; the Strix Halo Windows guide (#3029, first written by @HaiYunChuang, measured again here); a Polish README (#3355, by @j-luwierski, the 8th README).
- A graphics card that went through GPU resets can hang engine starts until the PC is rebooted. We saw it on a 4 x R9700 box; 40 loads after a fresh boot had no stall.

Names stay on the commits: 35 contributors' pull requests were ported with their authorship.

## What we did not test

- Prompts longer than a few thousand tokens at 512K or 1M DeepSeek contexts.
- RTX 40 cards (no Ada card here), A100-class cards, RX 6000 on Windows (#3895) and a two-GPU PC with a slow CPU (#3930): shipped from the reporters' measurements; please tell us.
- Whether the new summation orders (RTX PRO and Ampere-class cards) change answers: a 61-task check on the RTX PRO 5000 passed 54 against 55 with the switches off, and its differences stayed inside the spread between two identical runs; it is a small sample.
- The huge-page path with a reserved pool on the release package.
- The DeepSeek Strix Halo package on a 64 GB or 32 GB machine, and the 2.25 set on it.
- The route prior on the owner's RTX 5070: it applies there, and the quality check ran on the RTX 3060.
- The in-app updater from 0.2.6 to 0.2.7 (its files and signatures are checked; we run it after publishing).
- Any Mac beyond the signed build; Metal and SYCL got no 0.2.7 speed change.

## Updating

- **App:** the in-app updater installs the new installer. The app then offers the 0.1.52 engine. [TO CONFIRM with the update test]
- **Terminal:** run `strata update`. It fetches Control 0.1.52 and the newer engine. [TO CONFIRM with the update test]
- **DeepSeek engine:** updates itself on the next `strata launch`.

## Downloads

Three installers, the three Control packages (0.1.52), 13 Qwen engine packages and 4 DeepSeek engine packages (all 0.1.52: Windows CUDA 13, Windows AMD for Ryzen AI Max, Linux CUDA 13, Linux AMD rx9070), plus `strata-feed.json` with its signature, `latest.json`, `install-manifest.txt`, `install.sh`, `install.ps1` and `SHA256SUMS.txt`. The DeepSeek CUDA packages are built for sm_86, sm_89 and sm_120. [pending: f-027-gh-eng]

---

Strata is free and open source. If it runs well on your PC, a coffee keeps the work on it going:

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>
