import json
import shutil
import subprocess
from pathlib import Path

import pytest


DASHBOARD = Path("frontend/dashboard.html")


def _dashboard_import_source() -> str:
    html = DASHBOARD.read_text(encoding="utf-8")
    return html.split("// --- Import functions ---", maxsplit=1)[1].split(
        "let importPollTimer;",
        maxsplit=1,
    )[0]


def test_dashboard_import_flow_contains_preflight_confirmation():
    for rel in ("frontend/dashboard.html",):
        html = Path(rel).read_text(encoding="utf-8")

        assert 'id="import-preflight-panel"' in html
        assert 'id="import-start-confirm-btn"' in html
        assert "async function runImportPreflight(file)" in html
        assert "function renderImportPreflight" in html
        assert "/api/import/preflight" in html


def test_dashboard_uses_preflight_identity_snapshot_for_upload():
    html = DASHBOARD.read_text(encoding="utf-8")

    assert 'id="import-human-label"' in html
    assert 'id="import-assistant-label"' in html
    assert "仅用于 Echo Markdown 人物角色识别" in html
    assert "不会修改 Dashboard 身份或已有记忆" in html
    assert "let pendingImportSnapshot = null" in html
    assert "let importPreflightGeneration = 0" in html
    assert "await startImport(file, identityMapping)" in html
    assert "async function startImport(file, identityMapping)" in html

    upload_body = html.split(
        "async function startImport(file, identityMapping)",
        maxsplit=1,
    )[1].split("async function pauseImport", maxsplit=1)[0]
    assert "import-human-label" not in upload_body
    assert "import-assistant-label" not in upload_body


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_dashboard_import_snapshot_runtime_state_flow():
    script = r"""
const elements = new Map();
function makeElement(id) {
  return {
    id,
    value: '',
    checked: false,
    disabled: false,
    files: [],
    style: {},
    textContent: '',
    innerHTML: '',
    addEventListener() {},
    click() {},
  };
}
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  },
};
class FormData {
  constructor() { this.entries = []; }
  append(key, value) { this.entries.push([key, value]); }
}
const BASE = '';
function esc(value) { return String(value); }
const alerts = [];
function alert(message) { alerts.push(message); }
async function pollImportStatus() {}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return {promise, resolve, reject};
}
function preflightResponse(human, assistant) {
  return {
    async json() {
      return {
        ok: true,
        can_start: true,
        identity_mapping: {
          applied: true,
          human_label: human,
          assistant_label: assistant,
        },
      };
    },
  };
}

const fetchCalls = [];
const preflightRequests = [];
async function fetch(url, options) {
  fetchCalls.push({url, options});
  if (url.includes('/api/import/preflight')) {
    const request = deferred();
    preflightRequests.push(request);
    return request.promise;
  }
  if (url.includes('/api/import/upload')) {
    return {async json() { return {status: 'started'}; }};
  }
  throw new Error('unexpected fetch: ' + url);
}
""" + _dashboard_import_source() + r"""

(async function() {
  const humanInput = document.getElementById('import-human-label');
  const assistantInput = document.getElementById('import-assistant-label');

  const fileA = {name:'a.md', size:1, lastModified:1};
  const fileB = {name:'b.md', size:2, lastModified:2};
  humanInput.value = '松松';
  assistantInput.value = '尼奥';
  const pendingA = runImportPreflight(fileA);
  humanInput.value = '松松';
  assistantInput.value = '姐姐';
  const pendingB = runImportPreflight(fileB);
  preflightRequests[1].resolve(preflightResponse('松松', '姐姐'));
  await pendingB;
  preflightRequests[0].resolve(preflightResponse('松松', '尼奥'));
  await pendingA;
  const case1SnapshotIsB = (
    pendingImportFile === fileB
    && pendingImportSnapshot.file === fileB
    && pendingImportSnapshot.identity_mapping.assistant_label === '姐姐'
  );
  await confirmStartImport();
  const case1Upload = fetchCalls.filter(call => call.url.includes('/api/import/upload')).at(-1);
  const case1Params = new URL(case1Upload.url, 'http://localhost').searchParams;

  const fileC = {name:'c.md', size:3, lastModified:3};
  const pendingC = runImportPreflight(fileC);
  const requestC = preflightRequests.at(-1);
  clearImportPreflight();
  requestC.resolve(preflightResponse('松松', '尼奥'));
  await pendingC;
  const case2CancelledStayedEmpty = pendingImportFile === null && pendingImportSnapshot === null;

  const fileD = {name:'d.md', size:4, lastModified:4};
  humanInput.value = '松松';
  assistantInput.value = '尼奥';
  const pendingD = runImportPreflight(fileD);
  preflightRequests.at(-1).resolve(preflightResponse('松松', '尼奥'));
  await pendingD;
  humanInput.value = '另一个人';
  assistantInput.value = '另一个AI';
  await confirmStartImport();
  const case3Upload = fetchCalls.filter(call => call.url.includes('/api/import/upload')).at(-1);
  const case3Params = new URL(case3Upload.url, 'http://localhost').searchParams;

  const fileE = {name:'e.md', size:5, lastModified:5};
  const pendingE = runImportPreflight(fileE);
  preflightRequests.at(-1).resolve(preflightResponse('松松', '尼奥'));
  await pendingE;
  pendingImportFile = {name:'different.md', size:6, lastModified:6};
  const uploadsBeforeMismatch = fetchCalls.filter(call => call.url.includes('/api/import/upload')).length;
  await confirmStartImport();
  const uploadsAfterMismatch = fetchCalls.filter(call => call.url.includes('/api/import/upload')).length;

  process.stdout.write(JSON.stringify({
    case1SnapshotIsB,
    case1UploadHuman: case1Params.get('human_label'),
    case1UploadAssistant: case1Params.get('assistant_label'),
    case2CancelledStayedEmpty,
    case3UploadHuman: case3Params.get('human_label'),
    case3UploadAssistant: case3Params.get('assistant_label'),
    case4MismatchBlocked: uploadsBeforeMismatch === uploadsAfterMismatch,
    case4Alerted: alerts.some(message => message.includes('重新预检')),
  }));
})().catch(error => {
  console.error(error);
  process.exit(1);
});
"""
    completed = subprocess.run(
        [shutil.which("node"), "-e", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == {
        "case1SnapshotIsB": True,
        "case1UploadHuman": "松松",
        "case1UploadAssistant": "姐姐",
        "case2CancelledStayedEmpty": True,
        "case3UploadHuman": "松松",
        "case3UploadAssistant": "尼奥",
        "case4MismatchBlocked": True,
        "case4Alerted": True,
    }
