import puppeteer from "puppeteer-core";
// Chrome repro for the WebGPU fp16 zero-logit bug (see README.md).
//   npm install puppeteer-core && python3 -m http.server 8765 &  node probe.mjs [headful]
const headless = process.argv[2] !== "headful";
const models = [
  ["onnx-community/SmolLM-135M-ONNX", "f4acdf4c16b4deaee4cc6659fd3e32e44ecc37fd"],
  ["onnx-community/SmolLM2-135M-Instruct-ONNX", "b8a5c0f183b78c55955a5364f610c36668b5e681"],
  ["HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"],
];
const configs = [["wasm", "fp16"], ["webgpu", "fp32"], ["webgpu", "fp16"]];
const query = new URLSearchParams({ models: JSON.stringify(models), configs: JSON.stringify(configs) });
const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless, userDataDir: process.cwd() + "/.chrome-profile", protocolTimeout: 900000,
  args: ["--enable-unsafe-webgpu", "--no-first-run", "--no-default-browser-check"],
});
const page = await browser.newPage();
page.on("console", (m) => { if (m.text().startsWith("{")) console.log(m.text()); });
page.on("pageerror", (e) => console.log("pageerror", String(e)));
await page.goto(`http://localhost:8765/index.html?${query}`);
for (let i = 0; i < 180; i++) {
  if (await page.evaluate(() => window.done === true)) break;
  await new Promise((r) => setTimeout(r, 5000));
}
console.log(JSON.stringify({ chrome: await browser.version(), headless }));
await browser.close();
