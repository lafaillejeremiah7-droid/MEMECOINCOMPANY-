'use strict';

async function confirmShutdown({github, context, core, env = process.env,
  fetchImpl = fetch, wait = ms => new Promise(resolve => setTimeout(resolve, ms)), attempts = 72}) {
  let stopped = false;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const runs = await github.paginate(github.rest.actions.listWorkflowRuns, {
      ...context.repo, workflow_id: 'run-scanner.yml', per_page: 100
    });
    const active = runs.filter(run => String(run.id) !== String(context.runId) && run.status !== 'completed');
    if (active.length === 0) {
      stopped = true;
      break;
    }
    core.info(`Waiting for ${active.length} prior signal session(s) to stop.`);
    if (attempt + 1 < attempts) await wait(5000);
  }
  if (!stopped) throw new Error('Shutdown is not yet verified; no confirmation message was sent.');

  const token = env.MEMESCANNER_TELEGRAM_BOT_TOKEN?.trim();
  const chat = env.MEMESCANNER_TELEGRAM_CHAT_ID?.trim();
  if (!token || !chat) throw new Error('Company is OFF, but Telegram confirmation credentials are missing.');
  const text = [
    'Memecoin signal company (hii): OFF',
    'Confirmed: active and queued GitHub signal sessions have stopped.',
    'Automatic scanning and BUY/WATCH alerts are disabled until you request a restart.',
    `Confirmed at: ${new Date().toISOString()}`,
    'This is a shutdown confirmation, not a trade signal.'
  ].join('\n');
  let result;
  try {
    const response = await fetchImpl(`https://api.telegram.org/bot${token}/sendMessage`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({chat_id: chat, text, disable_web_page_preview: true}),
      signal: AbortSignal.timeout(15000)
    });
    result = await response.json();
    if (!response.ok || result?.ok !== true) throw new Error('Rejected');
  } catch {
    // No automatic resend: a transport failure can happen after acceptance.
    // Never include the credential-bearing URL or provider response in logs.
    throw new Error('Company is OFF, but Telegram delivery failed or is uncertain; no automatic resend.');
  }
  core.info('Company OFF verified; Telegram accepted the shutdown confirmation.');
  return true;
}

module.exports = {confirmShutdown};
