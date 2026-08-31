'use strict';

const SESSION_STEP = 'Run a 30-minute signal session';

function id(value) {
  if (!/^[1-9][0-9]*$/.test(String(value))) throw new Error('Invalid signal-history identifier');
  return BigInt(String(value));
}

function sessionStarted(jobs) {
  return jobs.some(job => (job.steps || []).some(step => {
    if (step.name !== SESSION_STEP) return false;
    if (step.conclusion === 'skipped') return false;
    if (['pending', 'queued'].includes(step.status) && step.conclusion == null) return false;
    // Unknown metadata is not proof of a safely skipped session.
    return true;
  }));
}

async function findPreviousSignalState({github, context, core}) {
  const {owner, repo} = context.repo;
  const current = id(context.runId);
  const artifacts = await github.paginate(github.rest.actions.listArtifacts,
    {owner, repo, name: 'signal-state', per_page: 100});
  const eligible = artifacts.filter(a => a.name === 'signal-state' &&
    a.workflow_run?.head_branch === 'main' && id(a.workflow_run.id) <= current);
  eligible.sort((a, b) => id(a.id) < id(b.id) ? 1 : -1);
  let prior = eligible[0];
  const runs = await github.paginate(github.rest.actions.listWorkflowRuns,
    {owner, repo, workflow_id: 'run-scanner.yml', per_page: 100});
  const newer = runs.filter(run => run.head_branch === 'main' && id(run.id) < current &&
    (!prior || id(run.id) > id(prior.workflow_run.id)));
  newer.sort((a, b) => id(a.id) < id(b.id) ? 1 : -1);
  for (const run of newer) {
    const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun,
      {owner, repo, run_id: run.id, per_page: 100});
    if (!sessionStarted(jobs)) continue;
    // Confirm against the specific run before diagnosing missing history: a
    // repository-wide artifact listing may not yet include a completed upload.
    const perRun = await github.paginate(github.rest.actions.listWorkflowRunArtifacts,
      {owner, repo, run_id: run.id, per_page: 100});
    const checkpoint = perRun.filter(a => a.name === 'signal-state' &&
      id(a.workflow_run?.id) === id(run.id)).sort((a, b) => id(a.id) < id(b.id) ? 1 : -1)[0];
    if (!checkpoint) {
      throw new Error(`Signal session ${run.id} started but has no checkpoint. Recover its history before restarting.`);
    }
    if (!prior || id(checkpoint.workflow_run.id) > id(prior.workflow_run.id)) prior = checkpoint;
  }
  if (prior?.expired) throw new Error('Signal state expired. Recover dedup history before restarting.');
  if (prior) {
    core.info(`Restoring signal history from completed checkpoint in run ${prior.workflow_run.id}.`);
    core.setOutput('run_id', String(prior.workflow_run.id));
  } else {
    core.notice('First signal-only deployment: new observation/dedup database.');
  }
  return prior ? String(prior.workflow_run.id) : null;
}

module.exports = {findPreviousSignalState, sessionStarted};
