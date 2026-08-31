'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {findPreviousSignalState, sessionStarted} = require('../scripts/signal_history.cjs');

const artifact = (run = 10, extra = {}) => ({id: run * 100, name: 'signal-state',
  workflow_run: {id: run, head_branch: 'main'}, expired: false, ...extra});
const job = (status = 'completed', conclusion = 'success') => ({steps: [
  {name: 'Run a 30-minute signal session', status, conclusion}
]});
function fixture({artifacts = [artifact()], runs = [], jobs = {}, perRun = {}, current = 30} = {}) {
  const queriedJobs = [], output = {};
  return {queriedJobs, output, args: {
    context: {repo: {owner: 'owner', repo: 'repo'}, runId: current},
    core: {info() {}, notice() {}, setOutput(k, v) {output[k] = v;}},
    github: {rest: {actions: {listArtifacts: 'artifacts', listWorkflowRuns: 'runs',
      listJobsForWorkflowRun: 'jobs', listWorkflowRunArtifacts: 'perRun'}},
      async paginate(method, params) {
        if (method === 'artifacts') return artifacts;
        if (method === 'runs') return runs;
        if (method === 'perRun') return perRun[params.run_id] || [];
        queriedJobs.push(params.run_id);
        return jobs[params.run_id] || [];
      }
    }
  }};
}

test('string and numeric run IDs exclude current and future sessions', async () => {
  const f = fixture({current: '30', runs: [{id: 30, head_branch: 'main'}, {id: 31, head_branch: 'main'}]});
  assert.equal(await findPreviousSignalState(f.args), '10');
  assert.deepEqual(f.queriedJobs, []);
});

for (const [status, conclusion] of [['pending', null], ['queued', null], ['completed', 'skipped']]) {
  test(`an unstarted ${status}/${conclusion} step does not discard a valid checkpoint`, async () => {
    const f = fixture({runs: [{id: 20, head_branch: 'main'}], jobs: {20: [job(status, conclusion)]}});
    assert.equal(await findPreviousSignalState(f.args), '10');
  });
}

for (const [status, conclusion] of [['in_progress', null], ['completed', 'failure'], ['completed', 'success'], [undefined, null]]) {
  test(`started or unknown ${status}/${conclusion} without checkpoint remains blocked`, async () => {
    const f = fixture({runs: [{id: 20, head_branch: 'main'}], jobs: {20: [job(status, conclusion)]}});
    await assert.rejects(findPreviousSignalState(f.args), /20 started but has no checkpoint/);
  });
}

test('a checkpoint confirmed on its exact run resolves listing lag', async () => {
  const f = fixture({runs: [{id: 20, head_branch: 'main'}], jobs: {20: [job()]}, perRun: {20: [artifact(20)]}});
  assert.equal(await findPreviousSignalState(f.args), '20');
  assert.equal(f.output.run_id, '20');
});

test('expired newest checkpoint never falls back to older history', async () => {
  const f = fixture({artifacts: [artifact(), artifact(20, {expired: true})]});
  await assert.rejects(findPreviousSignalState(f.args), /expired/);
});

test('an unrelated artifact or branch is never restored', async () => {
  const f = fixture({artifacts: [artifact(), artifact(20, {name: 'test-results'}),
    artifact(21, {workflow_run: {id: 21, head_branch: 'other'}})]});
  assert.equal(await findPreviousSignalState(f.args), '10');
});

test('genuinely first deployment can initialize without inventing history', async () => {
  const f = fixture({artifacts: []});
  assert.equal(await findPreviousSignalState(f.args), null);
  assert.equal(sessionStarted([job('completed', 'skipped')]), false);
});

test('rerunning an older workflow cannot roll back newer delivery history', async () => {
  const f = fixture({current: 20, artifacts: [artifact(10), artifact(30)],
    runs: [{id: 30, head_branch: 'main'}], jobs: {30: [job()]}});
  f.args.runAttempt = 2;
  await assert.rejects(findPreviousSignalState(f.args), /new workflow run/);
  assert.deepEqual(f.output, {});
});

test('deployment passes the real attempt number into the rerun guard', () => {
  const source = readFileSync('.github/workflows/run-scanner.yml', 'utf8');
  assert.match(source, /runAttempt: process\.env\.GITHUB_RUN_ATTEMPT/);
});
