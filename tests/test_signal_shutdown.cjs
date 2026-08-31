'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {confirmShutdown} = require('../scripts/signal_shutdown.cjs');

function fixture(states) {
  const sent = [], logs = [];
  return {sent, logs, args: {
    context: {repo: {owner: 'owner', repo: 'repo'}, runId: '30'},
    core: {info(message) {logs.push(message);}}, attempts: 3,
    env: {MEMESCANNER_TELEGRAM_BOT_TOKEN: 'test-token', MEMESCANNER_TELEGRAM_CHAT_ID: 'existing-chat'},
    github: {rest: {actions: {listWorkflowRuns: 'runs'}}, async paginate() {return states.length > 1 ? states.shift() : states[0];}},
    wait: async () => {},
    fetchImpl: async (url, options) => {sent.push(JSON.parse(options.body)); return {ok: true, json: async () => ({ok: true})};}
  }};
}

test('waits for active and queued sessions, then sends only to the configured chat', async () => {
  const f = fixture([[{id: 20, status: 'in_progress'}, {id: 21, status: 'pending'}],
    [{id: 20, status: 'completed'}, {id: 21, status: 'completed'}, {id: 30, status: 'in_progress'}]]);
  assert.equal(await confirmShutdown(f.args), true);
  assert.equal(f.sent.length, 1);
  assert.equal(f.sent[0].chat_id, 'existing-chat');
  assert.match(f.sent[0].text, /company \(hii\): OFF/);
  assert.equal(f.logs.filter(l => l.startsWith('Waiting')).length, 1);
});

test('never says OFF or sends confirmation while any old session remains active', async () => {
  const f = fixture([[{id: 20, status: 'in_progress'}]]);
  await assert.rejects(confirmShutdown(f.args), /not yet verified/);
  assert.deepEqual(f.sent, []);
});

test('does not guess a recipient when configured destination is missing', async () => {
  const f = fixture([[]]);
  delete f.args.env.MEMESCANNER_TELEGRAM_CHAT_ID;
  await assert.rejects(confirmShutdown(f.args), /credentials are missing/);
  assert.deepEqual(f.sent, []);
});

test('uncertain delivery is not retried and credentials are not exposed', async () => {
  const f = fixture([[]]);
  let calls = 0;
  f.args.fetchImpl = async () => {calls++; throw new Error('test-token private-url');};
  await assert.rejects(confirmShutdown(f.args), error => /uncertain/.test(error.message) && !/test-token|private-url/.test(error.message));
  assert.equal(calls, 1);
});

test('OFF workflow cancels the existing concurrency group and contains no scanner launch', () => {
  const source = readFileSync('.github/workflows/run-scanner.yml', 'utf8');
  assert.match(source, /group: signal-company-state/);
  assert.match(source, /cancel-in-progress: true/);
  assert.match(source, /confirmShutdown/);
  assert.doesNotMatch(source, /python -m memescanner|Run a 30-minute signal session|MEMESCANNER_TAVILY_API_KEY/);
});
