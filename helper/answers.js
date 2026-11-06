// Answers for the voice function (voice.js): Home Assistant's Assist, a language model through OpenRouter, and a server of yours. Text
// goes in, text comes out; the settings are the voice section of config.json (voice.js passes them in with init).
const fetch = require('./fetch'),
  hub = require('./toolhub'),
  { URL } = require('./netcompat');

let cfg = () => ({});
// deps.tools(): the tool providers, [{id, name, tools: {list(), call(name, args, opts)}}] (voice.js: the service's own and the add-ons')
let deps = { tools: () => [] };
const init = (f, d) => {
  cfg = f;
  if (d) deps = Object.assign({}, deps, d);
};

// POST JSON; resolves {status, text}. Redirects are not followed.
function request(url, headers, payload, ms, extra) {
  return fetch
    .read(url, {
      method: payload === undefined ? 'GET' : 'POST', // no payload: a GET
      headers,
      body: payload === undefined ? undefined : JSON.stringify(payload),
      idleMs: ms || 30000,
      redirects: 0,
      anyStatus: true,
      maxBytes: 1 << 20,
      bigText: 'the answer is too large',
      insecure: !!(extra && extra.rejectUnauthorized === false),
    })
    .then(r => ({ status: r.status, text: r.body.toString('utf8') }))
    .catch(e => {
      if (e.code === 'timeout') throw new Error('no answer in time');
      if (e.code === 'url')
        throw new Error(/http/.test(e.message) ? 'the address must start with http:// or https://' : 'the address is not valid');
      throw e;
    });
}
// a server of yours: POST {"text": ...}; the answer may be JSON ({"text", "title", "image", "seconds"}) or plain text
async function toServer(c, payload) {
  const r = await request(c.url, c.token ? { authorization: 'Bearer ' + c.token } : {}, payload, 30000);
  if (r.status < 200 || r.status >= 300) throw new Error('your server answered ' + r.status);
  let o = null;
  try {
    o = JSON.parse(r.text);
  } catch (_) {}
  if (o && typeof o === 'object') {
    const text = String(o.text || o.reply || o.answer || o.message || '').trim();
    return text || o.image ? { title: o.title ? String(o.title) : '', text, image: o.image, seconds: o.seconds } : null;
  }
  const t = r.text.trim();
  return t ? { text: t } : null;
}

// ---- chat with a language model through OpenRouter (text only; the key and the questions go to OpenRouter) ----
const orApi = () => process.env.CYTV_OPENROUTER || 'https://openrouter.ai/api/v1'; // overridden only by the tests
const LLM_DEFAULTS = { enabled: false, key: '', model: 'openrouter/auto', system: '', agent: false };
const llmCfg = () => Object.assign({}, LLM_DEFAULTS, cfg().llm || {});
const DEFAULT_SYSTEM =
  'You are a voice assistant on a television. You have no tools and no internet access: answer from your own knowledge, and say when something may be out of date. The user speaks to you through the remote control and your answer appears as text on the screen, so keep it short: one to three plain sentences, no markdown, no lists, no emoji. If the speech looks like a recognition mistake, answer what was most likely meant.';
let history = [],
  lastTurn = 0;
const TOOLISH = /<\|?\s*(?:tool|function|python_tag)|<tool_call|\[TOOL_CALL|<function_call/i; // a model that tries to call a tool instead of answering
function cleanAnswer(t) {
  let x = String(t || '').replace(/<think>[\s\S]*?<\/think>/gi, '');
  const m = TOOLISH.exec(x);
  if (m) x = x.slice(0, m.index);
  return x
    .replace(/<\|[^|>]*\|>/g, '')
    .replace(/\s+\n/g, '\n')
    .trim();
}
const orHeaders = key => ({
  authorization: 'Bearer ' + key,
  'HTTP-Referer': 'https://github.com/lasharor/cleanyourtv-WebOS',
  'X-Title': 'CleanYourTV',
});
const today = () => ' Today is ' + new Date().toISOString().slice(0, 10) + '.';
async function askLlm(text, opts) {
  const l = llmCfg();
  if (!l.key) throw new Error('no OpenRouter key is set');
  if (Date.now() - lastTurn > 600000) history = []; // a new conversation after ten quiet minutes
  const turns = history.concat({ role: 'user', content: text }).slice(-12);
  if (l.agent) {
    const tools = await gatherTools();
    if (tools.names.length) {
      const r = await agentAsk(l, turns, tools, opts || {});
      if (r !== null) return r;
    }
  }
  return plainAsk(l, turns);
}
// today's way: one answer from the model's own knowledge, no tools
async function plainAsk(l, turns) {
  const system = (l.system || DEFAULT_SYSTEM) + today();
  let answer = '',
    tool = false;
  for (let attempt = 0; attempt < 2 && !answer; attempt++) {
    const messages = [
      {
        role: 'system',
        content:
          system +
          (attempt
            ? ' You have no tools and cannot search the internet: never call a tool. Answer in plain text from what you know, and say so if it may be out of date.'
            : ''),
      },
    ].concat(turns);
    const r = await request(
      orApi() + '/chat/completions',
      orHeaders(l.key),
      { model: l.model, messages, max_tokens: 500, reasoning: { exclude: true } },
      60000
    );
    let o = {};
    try {
      o = JSON.parse(r.text);
    } catch (_) {}
    if (r.status !== 200 || o.error)
      throw new Error('OpenRouter: ' + String((o.error && (o.error.message || o.error.code)) || 'HTTP ' + r.status).slice(0, 160));
    const msg = (o.choices && o.choices[0] && o.choices[0].message) || {},
      raw = String(msg.content || '');
    tool = tool || TOOLISH.test(raw) || !!(msg.tool_calls && msg.tool_calls.length);
    answer = cleanAnswer(raw);
    if (tool && !attempt) answer = ''; // retry once, telling the model it has no tools
  }
  if (!answer)
    throw new Error(
      tool ? 'The model tried to use a tool instead of answering. Choose another model.' : 'OpenRouter returned an empty answer'
    );
  history = turns.concat({ role: 'assistant', content: answer });
  lastTurn = Date.now();
  return answer;
}

// ---- the agent: the model may use the tools the service and the approved add-ons give (llm.agent) ----
const MAX_ROUNDS = 5, // every tool goes with every question (at most toolhub.MAX_TOOLS): they cost prompt tokens on each round
  MAX_RESULT = 2000,
  CONFIRM_MS = 30000,
  TOOL_MS = 30000;
const AGENT_SYSTEM =
  'You are a voice assistant on a television, and you can act on the TV with the tools you are given. Use a tool when the user asks you to do something on the TV, or asks about something a tool can look up; otherwise answer from your own knowledge, and say when something may be out of date. The user speaks to you through the remote control and your answer appears as text on the screen, so keep it short: one to three plain sentences, no markdown, no lists, no emoji. After using a tool, say in one short sentence what you did or found; if a tool fails, say so plainly. If the speech looks like a recognition mistake, answer what was most likely meant.';
const YES = /^(yes|yeah|yep|yup|sure|ok|okay|do it|go ahead|confirm)\b/i;
let pending = null; // {provider, name, args, description, at}: a tool that waits for the person's yes

// the tools of every provider (toolhub.merge: the same names /api/mcp lists), as OpenRouter takes them
async function gatherTools() {
  let providers = [];
  try {
    providers = deps.tools() || [];
  } catch (_) {}
  const m = await hub.merge(providers);
  m.defs = m.names.map(n => ({
    type: 'function',
    function: { name: n, description: m.map[n].description, parameters: m.map[n].schema },
  }));
  return m;
}

// what a tool answered, as text for the model, cut short
function resultText(r) {
  let s;
  if (r && Array.isArray(r.content))
    s = r.content
      .filter(c => c && c.type === 'text')
      .map(c => String(c.text))
      .join('\n');
  else s = typeof r === 'string' ? r : JSON.stringify(r === undefined ? null : r);
  s = (r && r.isError ? 'Error: ' : '') + (s || '');
  return s.length > MAX_RESULT ? s.slice(0, MAX_RESULT) + '…' : s;
}
function callTool(entry, args, confirmed) {
  let timer;
  const call = Promise.resolve().then(() => entry.provider.tools.call(entry.name, args, confirmed ? { confirmed: true } : {}));
  const late = new Promise(resolve => {
    timer = setTimeout(() => resolve({ content: [{ type: 'text', text: 'the tool did not answer in time' }], isError: true }), TOOL_MS);
  });
  return Promise.race([call, late])
    .catch(e => ({ content: [{ type: 'text', text: String((e && e.message) || e) }], isError: true }))
    .then(r => {
      clearTimeout(timer);
      return r;
    });
}
// the question the TV asks before a tool that turns something off or deletes something
const confirmText = d => String(d).replace(/[.\s]+$/, '') + '? Say yes within 30 seconds to go ahead.';

// the loop: send the tools, run the calls the model asks for, give it the results, until it answers in words.
// null: this model cannot use tools, so the caller asks without them
async function agentAsk(l, turns, tools, opts) {
  const system = (l.system ? l.system + ' You can act on the TV with the tools you are given.' : AGENT_SYSTEM) + today();
  const messages = [{ role: 'system', content: system }].concat(turns);
  for (let round = 0; round < MAX_ROUNDS; round++) {
    const r = await request(
      orApi() + '/chat/completions',
      orHeaders(l.key),
      { model: l.model, messages, tools: tools.defs, tool_choice: 'auto', max_tokens: 500, reasoning: { exclude: true } },
      60000
    );
    let o = {};
    try {
      o = JSON.parse(r.text);
    } catch (_) {}
    if (r.status !== 200 || o.error) {
      const m = String((o.error && (o.error.message || o.error.code)) || 'HTTP ' + r.status);
      if (!round && /tool/i.test(m) && (r.status === 404 || r.status === 400)) return null; // no endpoint of this model takes tools
      throw new Error('OpenRouter: ' + m.slice(0, 160));
    }
    const msg = (o.choices && o.choices[0] && o.choices[0].message) || {},
      calls = Array.isArray(msg.tool_calls) ? msg.tool_calls.filter(c => c && c.function) : [];
    if (!calls.length) {
      const raw = String(msg.content || '');
      if (!round && TOOLISH.test(raw)) return null; // it wrote a tool call as text: it cannot really use tools
      const answer = cleanAnswer(raw);
      if (!answer) throw new Error('OpenRouter returned an empty answer');
      history = turns.concat({ role: 'assistant', content: answer });
      lastTurn = Date.now();
      return answer;
    }
    messages.push({ role: 'assistant', content: msg.content || null, tool_calls: calls });
    let ask = null;
    for (const c of calls) {
      const name = String(c.function.name || ''),
        entry = tools.map[name];
      let args = null,
        out;
      try {
        args = JSON.parse(c.function.arguments || '{}');
      } catch (_) {}
      const bad = entry ? hub.checkArgs(entry.schema, args) : '';
      if (!entry) out = 'Error: there is no tool named ' + name.slice(0, 64);
      else if (bad) out = 'Error: ' + bad;
      else if (entry.confirm) {
        // never run here: the person says yes on the TV first (confirmPending)
        if (!ask) ask = { provider: entry.provider, name: entry.name, args, description: entry.description, at: Date.now() };
        out = 'Not run: this tool needs the person to say yes first, and the TV is asking them now.';
      } else {
        if (opts.onTool) opts.onTool(name);
        out = resultText(await callTool(entry, args, false));
      }
      messages.push({ role: 'tool', tool_call_id: String(c.id || name), content: out });
    }
    if (ask) {
      pending = ask;
      const answer = confirmText(ask.description);
      history = turns.concat({ role: 'assistant', content: answer });
      lastTurn = Date.now();
      return answer;
    }
  }
  throw new Error('the assistant used too many tools without answering');
}

// a tool waiting for a yes: whether the words answer it. null: nothing waits (or it waited too long), so the words are a new question
function pendingConfirm() {
  if (pending && Date.now() - pending.at > CONFIRM_MS) pending = null;
  return pending ? { name: pending.name, description: pending.description } : null;
}
async function confirmPending(text) {
  const p = pendingConfirm();
  if (!p) return null;
  const run = pending;
  pending = null;
  let answer;
  if (!YES.test(String(text || '').trim())) answer = 'Okay, I did not do that.';
  else {
    const r = await callTool(run, run.args, true);
    answer =
      r && r.isError
        ? 'That did not work: ' +
          resultText(r)
            .replace(/^Error: /, '')
            .slice(0, 200)
        : 'Done.';
  }
  history = history.concat({ role: 'user', content: String(text) }, { role: 'assistant', content: answer }).slice(-12);
  lastTurn = Date.now();
  return answer;
}

// ---- Home Assistant: the Assist (conversation) API with a long-lived access token ----
const HA_DEFAULTS = { url: '', token: '', selfSigned: false, route: 'ha-first', agent: '' };
const HA_ROUTES = ['ha-first', 'ha-only', 'ai-only'];
const haCfg = () => Object.assign({}, HA_DEFAULTS, cfg().ha || {});
const haReady = h => !!(h.url && h.token);
let haConv = null; // { id, at }: Assist keeps the context of a conversation; a new one starts after ten quiet minutes
// the address without credentials, query or trailing slash; '' when it is not an http(s) address
function haBase(t) {
  let u;
  try {
    u = new URL(String(t).trim());
  } catch (_) {
    return '';
  }
  return u.protocol === 'http:' || u.protocol === 'https:' ? u.protocol + '//' + u.host + u.pathname.replace(/\/+$/, '') : '';
}
// Assist asks for the same text twice only if it did not understand: error code no_intent_match
async function haCall(h, path, payload, ms) {
  const r = await request(
    h.url + path,
    { authorization: 'Bearer ' + h.token },
    payload,
    ms || 30000,
    h.selfSigned && /^https:/.test(h.url) ? { rejectUnauthorized: false } : {}
  );
  if (r.status === 401 || r.status === 403) throw new Error('Home Assistant refused the access token');
  if (r.status === 404) throw new Error('that does not look like Home Assistant (' + path + ' was not found)');
  if (r.status < 200 || r.status >= 300) throw new Error('Home Assistant answered ' + r.status);
  try {
    return JSON.parse(r.text);
  } catch (_) {
    throw new Error('Home Assistant did not answer with JSON');
  }
}
async function askHa(text) {
  const h = haCfg();
  if (!haReady(h)) throw new Error('Home Assistant is not set up');
  if (haConv && Date.now() - haConv.at > 600000) haConv = null;
  const body = { text, language: 'en' };
  if (haConv) body.conversation_id = haConv.id;
  if (h.agent) body.agent_id = h.agent;
  const o = await haCall(h, '/api/conversation/process', body, 30000),
    res = (o && o.response) || {};
  const type = String(res.response_type || ''),
    code = String((res.data && res.data.code) || '');
  const speech = String((res.speech && res.speech.plain && res.speech.plain.speech) || '').trim();
  if (o.conversation_id) haConv = { id: String(o.conversation_id), at: Date.now() };
  return {
    text: speech || (type === 'action_done' ? 'Done' : ''),
    understood: !(type === 'error' && code === 'no_intent_match'),
    type,
    code,
  };
}
async function haTest() {
  const h = haCfg();
  if (!haReady(h)) throw new Error('set the Home Assistant address and token first');
  const c = await haCall(h, '/api/config', undefined, 10000);
  return { name: String(c.location_name || ''), version: String(c.version || '') };
}
// conversation agents show up as conversation.* entities
async function haAgents() {
  const h = haCfg();
  if (!haReady(h)) throw new Error('set the Home Assistant address and token first');
  const all = await haCall(h, '/api/states', undefined, 20000);
  return (Array.isArray(all) ? all : [])
    .filter(e => e && /^conversation\.[\w]+$/.test(e.entity_id || ''))
    .map(e => ({ id: e.entity_id, name: String((e.attributes && e.attributes.friendly_name) || e.entity_id).slice(0, 80) }));
}

// ---- models of OpenRouter: a short list for the TV, everything for the phone ----
let modelCache = null;
const MAJOR = ['anthropic', 'openai', 'google', 'meta-llama', 'mistralai', 'deepseek', 'qwen', 'x-ai'];
function models() {
  if (modelCache && Date.now() - modelCache.at < 3600000) return Promise.resolve(modelCache);
  return fetch
    .read(orApi() + '/models', {
      headers: { 'User-Agent': 'cleanyourtv-voice' },
      idleMs: 15000,
      maxBytes: 8 << 20,
      bigText: 'the model list is too large',
      statusText: st => 'OpenRouter answered ' + st,
    })
    .catch(e => {
      throw e.code === 'timeout' ? new Error('OpenRouter did not answer') : e;
    })
    .then(r => {
      let all;
      try {
        all = JSON.parse(r.body.toString('utf8'))
          .data.filter(
            m =>
              m &&
              /^[\w.\-:\/~]{1,100}$/.test(m.id || '') &&
              !(
                m.architecture &&
                Array.isArray(m.architecture.output_modalities) &&
                m.architecture.output_modalities.some(x => x !== 'text')
              )
          )
          .map(m => ({
            id: m.id,
            name: String(m.name || m.id).slice(0, 80),
            created: m.created || 0,
            price: m.pricing && +m.pricing.prompt >= 0 ? Math.round(+m.pricing.prompt * 1e8) / 100 : null, // dollars per million input tokens
            tools: Array.isArray(m.supported_parameters) && m.supported_parameters.indexOf('tools') >= 0, // it can act on the TV (llm.agent)
          }));
      } catch (e) {
        throw new Error('the model list could not be read');
      }
      const picks = [],
        seen = {};
      for (const m of all.slice().sort((a, b) => b.created - a.created)) {
        const prov = m.id.split('/')[0];
        if (m.id.includes(':') || !MAJOR.includes(prov)) continue;
        if ((seen[prov] = (seen[prov] || 0) + 1) <= 3) picks.push(m);
      }
      return (modelCache = { at: Date.now(), all: all.sort((a, b) => a.id.localeCompare(b.id)), picks });
    });
}

// a new conversation: forget what was said to the model and Assist's conversation
function forget() {
  history = [];
  haConv = null;
  pending = null;
}
const forgetHa = () => {
  haConv = null;
};

module.exports = {
  init,
  toServer,
  askLlm,
  confirmPending,
  pendingConfirm,
  askHa,
  haTest,
  haAgents,
  models,
  forget,
  forgetHa,
  haBase,
  haReady,
  llmCfg,
  haCfg,
  LLM_DEFAULTS,
  HA_DEFAULTS,
  HA_ROUTES,
};
