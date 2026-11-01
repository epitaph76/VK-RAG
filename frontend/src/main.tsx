import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ReactMarkdown from 'react-markdown';
import { ArrowUp, ArrowUpRight, BookOpen, Check, ChevronDown, ChevronRight, Database, FileText, FlaskConical, Globe2, Layers3, LoaderCircle, LockKeyhole, LogOut, MessageSquare, Plus, RefreshCw, Search, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Upload, X } from 'lucide-react';
import './style.css';

type Source = { id: string; name: string; kind: string; url?: string; enabled?: number; last_run?: string; last_error?: string };
type Hit = { id: string; title: string; url: string; text: string; published_at: string | null; source_name: string; retrieval_score: number; rerank_score: number | null; aliases: { title: string; url: string }[] };
type Meta = { warnings: string[]; rerank_applied: boolean; gate: { decision: string; reason: string; confidence: number | null }; timings_ms: Record<string, number>; index_version: string | null; llm_error?: string; generation?: { requested: boolean; attempted: boolean; status: string; fallback_reason: string | null } };
type Result = { answer: string; sources: Hit[]; answer_mode: string; meta: Meta; model?: string };
type Turn = { id: string; question: string; result: Result };
type Job = { id: string; kind: string; state: string; progress: number; total: number; created_at: string; error?: string; result: { phase?: string; current_source?: string; current?: { retries?: number; retry_events?: { attempt: number; max_attempts: number; reason: string; delay_seconds: number }[] }; index_reused?: boolean; errors?: { source: string; error: string }[]; sources?: { source: string; fetched: number; changed: number; skipped: number; retries?: number }[]; index?: { version: string; warnings: string[] } } };
type Status = { documents: number; revision: number; index: { chunks: number; dense: boolean; version: string; revision: number; warnings: string[] } | null; llm: { configured: boolean; model: string }; models: { loaded: string[]; errors: Record<string, string> }; limits: { ask_per_minute: number } };
type Compared = Record<string, { hits: Hit[]; meta: Meta }>;

let csrf = '';
async function api(path: string, options: RequestInit = {}) {
  const headers: Record<string, string> = { 'X-CSRF-Token': csrf };
  if (options.body && !(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';
  const response = await fetch('/api' + path, { ...options, headers: { ...headers, ...options.headers } });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Проверьте заполненные поля');
  return data;
}
const when = (value?: string | null) => value ? new Date(value).toLocaleDateString('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' }) : 'Дата не указана';
const sourceLink = (url: string) => url.startsWith('/documents/') ? '/api' + url : url;
function GenerationNotice({ result }: { result: Result }) {
  if (result.answer_mode === 'llm') return null;
  const generation = result.meta.generation;
  // Earlier browser history already contains llm_error, even before this notice existed.
  const reason = generation?.fallback_reason || result.meta.llm_error ||
    (result.meta.warnings.some(w => w.includes('LLM не подключена')) ? 'not_configured' : null);
  if (!reason) return null;
  const descriptions: Record<string, string> = {
    timeout: 'Сервис генерации не ответил вовремя.',
    invalid_json: 'Модель вернула ответ в некорректном формате.',
    invalid_citations: 'Ссылки в ответе модели не прошли проверку.',
    invalid_semantic_sources: 'Модель некорректно указала источники ответа.',
    invalid_semantic_status: 'Модель вернула некорректный результат отбора источников.',
    empty_response: 'Сервис генерации вернул пустой ответ.',
    invalid_response_or_connection: 'Не удалось получить корректный ответ от сервиса генерации.',
    not_configured: 'Сервис генерации не подключён на сервере.',
    user_disabled: 'Генерация была выключена в параметрах этого вопроса.',
  };
  const description = /^http_\d{3}$/.test(reason) ? `Сервис генерации вернул HTTP ${reason.slice(5)}.` :
    descriptions[reason] || 'Генерация не завершилась.';
  const automatic = reason !== 'user_disabled';
  return <div className={'generation-notice ' + (automatic ? 'fallback' : 'disabled')} role="status">
    <strong>{automatic ? 'Автоматический переход к выдержкам' : 'Режим без генерации'}</strong>
    <span>{description} {automatic ? (result.sources.length ? 'Ниже показаны фрагменты найденных материалов.' : 'Локальных фрагментов для ответа недостаточно.') : ''}</span>
    {automatic && <span>Настройка «Генерация ответа» не изменялась.</span>}
  </div>;
}
function JobCard({ job }: { job: Job }) {
  // Older jobs used `done` even when only some sources were updated.
  const state = job.state === 'done' && job.error ? 'partial' : job.state;
  const labels: Record<string, string> = { queued: 'В очереди', running: 'Выполняется', done: 'Готово', partial: 'Частично', failed: 'Ошибка' };
  const lastRetry = job.result.current?.retry_events?.at(-1);
  return <div className="job">
    <span className={'job-state ' + state}>{labels[state] || state}</span>
    <strong>{job.kind === 'index' || job.result.phase === 'index' ? 'Построение индекса' : 'Сбор материалов'}</strong>
    <span className="muted">{job.progress}/{job.total}{job.result.phase === 'collect' ? ' страниц' : job.result.index || job.result.phase === 'index' ? ' фрагментов в индексе' : ''}</span>
    {state === 'running' && <LoaderCircle className="spin" size={14}/>}
    <span className="job-date">{when(job.created_at)}</span>
    {job.error && <p className="job-error">{job.error}</p>}
    {state === 'running' && job.result.current_source && <p className="muted">Источник: {job.result.current_source}.{lastRetry ? ` Повторов запросов: ${job.result.current?.retries}. Последний: ${lastRetry.reason}.` : ''}</p>}
    {job.result.index_reused && <p className="muted">Данные не изменились, использован готовый индекс.</p>}
    {(job.result.sources?.length || job.result.errors?.length) ? <div className="job-sources">
      {job.result.sources?.map(source => <p key={source.source}>
        <strong>{source.source}</strong>: загружено {source.fetched}, обновлено {source.changed}{source.skipped ? `, пропущено ${source.skipped}` : ''}{source.retries ? `, повторов запросов: ${source.retries}` : ''}.
      </p>)}
      {job.result.errors?.map((error, index) => <p className="job-error" key={index}><strong>{error.source}</strong>: {error.error}</p>)}
    </div> : null}
    <details><summary>Технические подробности</summary><pre>{JSON.stringify(job.result, null, 2)}</pre></details>
  </div>;
}
const referencedAnswer = (result: Result) => result.answer.replace(/\[(\d+)\](?:\([^)]*\))?/g, (original, number) => {
  const hit = result.sources[Number(number) - 1];
  return hit ? '[' + number + '](' + sourceLink(hit.url) + ')' : original;
});
const suggestions = ['Что нового в искусственном интеллекте?', 'Как технологии помогают научным исследованиям?', 'Какие открытия сделали астрономы?'];

function App() {
  const [view, setView] = useState<'chat' | 'admin' | 'lab'>('chat');
  const [sources, setSources] = useState<Source[]>([]);
  const [turns, setTurns] = useState<Turn[]>(() => { try { return JSON.parse(sessionStorage.getItem('srez-chat') || '[]'); } catch { return []; } });
  const [selected, setSelected] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [admin, setAdmin] = useState(false);
  const [password, setPassword] = useState('');
  const [filters, setFilters] = useState(false);
  const [sourceId, setSourceId] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [useLlm, setUseLlm] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [status, setStatus] = useState<Status | null>(null);
  const [adminSources, setAdminSources] = useState<Source[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [notice, setNotice] = useState('');
  const [sourceName, setSourceName] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [sourceKind, setSourceKind] = useState('rss');
  const [compared, setCompared] = useState<Compared | null>(null);
  const [labQuery, setLabQuery] = useState('');
  const [adminBusy, setAdminBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const active = turns.find(t => t.id === selected) || turns[turns.length - 1];
  const selectedSources = active?.result.sources || [];

  const refresh = async () => {
    setSources(await api('/sources'));
    if (csrf) {
      const [s, list, j] = await Promise.all([api('/admin/status'), api('/admin/sources'), api('/admin/jobs')]);
      setStatus(s); setAdminSources(list); setJobs(j);
    }
  };
  useEffect(() => {
    api('/sources').then(setSources).catch(() => setError('Сервер недоступен. Попробуйте обновить страницу.'));
    api('/admin/session').then(s => { csrf = s.csrf; setAdmin(true); }).catch(() => {});
  }, []);
  useEffect(() => { try { sessionStorage.setItem('srez-chat', JSON.stringify(turns)); } catch { /* quota: keep the active in-memory chat */ } }, [turns]);
  useEffect(() => { if (admin) refresh().catch(e => setError(e.message)); }, [admin, view]);
  useEffect(() => {
    if (!admin || !jobs.some(j => ['queued', 'running'].includes(j.state))) return;
    const timer = setInterval(() => refresh().catch(e => setError(e.message)), 3000);
    return () => clearInterval(timer);
  }, [admin, jobs]);
  useEffect(() => { if (turns.length) bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [turns.length, busy]);

  async function ask(text = query) {
    if (!text.trim() || busy) return;
    setBusy(true); setError(''); setQuery('');
    try {
      const result = await api('/ask', { method: 'POST', body: JSON.stringify({ q: text.trim(), rerank: 'auto', use_llm: useLlm, source_id: sourceId || null, date_from: dateFrom || null, date_to: dateTo || null }) });
      const turn = { id: crypto.randomUUID(), question: text.trim(), result };
      setTurns(old => [...old, turn]); setSelected(turn.id);
    } catch (e) { setQuery(text); setError((e as Error).message); }
    finally { setBusy(false); }
  }
  async function login(e: React.FormEvent) {
    e.preventDefault(); setError(''); setAdminBusy(true);
    try { const data = await api('/admin/login', { method: 'POST', body: JSON.stringify({ password }) }); csrf = data.csrf; setPassword(''); setAdmin(true); }
    catch (e) { setError((e as Error).message); } finally { setAdminBusy(false); }
  }
  async function action(fn: () => Promise<unknown>, message: string) {
    setAdminBusy(true); setError(''); setNotice('');
    try { await fn(); setNotice(message); await refresh(); } catch (e) { setError((e as Error).message); } finally { setAdminBusy(false); }
  }
  async function compare(e: React.FormEvent) {
    e.preventDefault(); setAdminBusy(true); setError('');
    try { setCompared(await api('/admin/compare', { method: 'POST', body: JSON.stringify({ q: labQuery, limit: 10, source_id: sourceId || null, date_from: dateFrom || null, date_to: dateTo || null }) })); }
    catch (e) { setError((e as Error).message); } finally { setAdminBusy(false); }
  }

  return <div className="app-shell">
    <aside className="rail">
      <a className="logo" href="#" onClick={e => { e.preventDefault(); setView('chat'); }} aria-label="Срез — главная"><span className="logo-mark"><Layers3 size={23}/></span><span>срез<span className="logo-dot">.</span></span></a>
      <div className="rail-label">РАБОЧЕЕ ПРОСТРАНСТВО</div>
      <nav><button className={view === 'chat' ? 'nav active' : 'nav'} onClick={() => setView('chat')}><MessageSquare size={18}/>Диалог<span className="nav-hint">01</span></button><button className={view === 'lab' ? 'nav active' : 'nav'} onClick={() => { setView('lab'); setError(''); }}><FlaskConical size={18}/>Лаборатория<LockKeyhole size={12}/></button><button className={view === 'admin' ? 'nav active' : 'nav'} onClick={() => { setView('admin'); setError(''); }}><Database size={18}/>База знаний<LockKeyhole size={12}/></button></nav>
      <div className="rail-label sources-label">ИСТОЧНИКИ <span>{sources.length.toString().padStart(2, '0')}</span></div>
      <div className="rail-sources">{sources.map(s => <button key={s.id} onClick={() => { setSourceId(sourceId === s.id ? '' : s.id); setView('chat'); setFilters(true); }} className={sourceId === s.id ? 'source-nav chosen' : 'source-nav'}><span className="tiny-icon">{s.kind === 'upload' ? <FileText size={14}/> : <Globe2 size={14}/>}</span><span>{s.name}</span><span className="source-dot"/></button>)}</div>
      <div className="rail-bottom"><span className="small-graphic"><BookOpen size={20}/></span><strong>Проверяемые ответы</strong><p>Каждый вывод начинается<br/>с конкретного материала.</p><div className="rail-foot">SITE RAG <span>V1.0</span></div></div>
    </aside>
    <div className="workspace">
      <header className="topbar"><div className="breadcrumb">Рабочее пространство<ChevronRight size={13}/><strong>{view === 'chat' ? 'Диалог' : view === 'lab' ? 'Лаборатория' : 'База знаний'}</strong></div><div className="header-actions"><span className="status-pill"><span/>С опорой на источники</span><button className="icon-button" aria-label="Управление базой" onClick={() => setView('admin')}><Settings2 size={18}/></button><span className="avatar">{admin ? 'A' : 'Г'}</span></div></header>
      {error && <div role="alert" className="alert error">{error}<button aria-label="Закрыть ошибку" onClick={() => setError('')}><X size={15}/></button></div>}
      {view === 'chat' ? <div className="chat-layout">
        <main className="chat-main">
          {!turns.length && !busy ? <div className="welcome"><div className="eyebrow"><span/> ВАШИ МАТЕРИАЛЫ. НОВЫЕ СВЯЗИ.</div><h1>Сначала источники.<br/><span>Потом ответы.</span></h1><p className="intro">Задайте вопрос своей базе знаний.<br/>Срез найдёт главное и покажет, откуда взялся ответ.</p><div className="suggestions"><div className="suggestion-label">С ЧЕГО НАЧАТЬ</div>{suggestions.map((s, i) => <button key={s} onClick={() => ask(s)}><span className="suggestion-number">0{i + 1}</span>{s}<ArrowUpRight size={17}/></button>)}</div><div className="how"><span><Search size={14}/>Находим фрагменты</span><i/><span><Sparkles size={14}/>Собираем ответ</span><i/><span><BookOpen size={14}/>Показываем источники</span></div></div> : <div className="conversation"><div className="conversation-top"><span className="eyebrow">ДИАЛОГ С МАТЕРИАЛАМИ</span><button onClick={() => { setTurns([]); setSelected(null); setError(''); }} disabled={busy}><Plus size={14}/>Новый диалог</button></div>{turns.map(turn => <article key={turn.id} className={active?.id === turn.id ? 'turn selected-turn' : 'turn'} onClick={() => setSelected(turn.id)}><div className="question"><span className="question-marker">ВЫ</span><h2>{turn.question}</h2></div><div className="answer-label"><span className="answer-mark"><Layers3 size={15}/></span>СРЕЗ<span className="mode-tag">{turn.result.answer_mode === 'llm' ? 'ОТВЕТ ПО ИСТОЧНИКАМ' : 'ВЫДЕРЖКИ ИЗ МАТЕРИАЛОВ'}</span></div><GenerationNotice result={turn.result}/><div className="answer-text"><ReactMarkdown components={{ img: () => null, a: ({ href, children }) => <a href={href} target="_blank" rel="noreferrer">{children}</a> }}>{referencedAnswer(turn.result)}</ReactMarkdown></div><div className="answer-footer"><button onClick={() => { setSelected(turn.id); document.getElementById('sources-panel')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }}><BookOpen size={14}/>{turn.result.sources.length} источников<ArrowUpRight size={13}/></button><span>{((turn.result.meta.timings_ms.request_total || turn.result.meta.timings_ms.total) / 1000).toFixed(1)} с</span></div>{turn.result.meta.warnings.length > 0 && <details className="warnings"><summary>Сведения об обработке</summary>{turn.result.meta.warnings.map(w => <p key={w}>{w}</p>)}</details>}</article>)}{busy && <div className="thinking" role="status"><LoaderCircle className="spin" size={19}/><div><strong>Сопоставляем материалы</strong><span>Первый запрос может занять больше времени: загружаются модели.</span></div></div>}<div ref={bottom}/></div>}
          <div className="composer-area">{filters && <div className="filters"><label>Источник<select value={sourceId} onChange={e => setSourceId(e.target.value)}><option value="">Все источники</option>{sources.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}</select></label><label>С даты<input type="date" value={dateFrom} onChange={e => setDateFrom(e.target.value)}/></label><label>По дату<input type="date" value={dateTo} onChange={e => setDateTo(e.target.value)}/></label><label className="checkbox"><input type="checkbox" checked={useLlm} onChange={e => setUseLlm(e.target.checked)}/>Генерация ответа</label></div>}<form className="composer" onSubmit={e => { e.preventDefault(); ask(); }}><textarea aria-label="Ваш вопрос" placeholder="Что вы хотите узнать?" rows={2} value={query} maxLength={1000} disabled={busy} onChange={e => setQuery(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } }}/><div className="composer-bottom"><button type="button" className={filters ? 'filter-button enabled' : 'filter-button'} onClick={() => setFilters(!filters)}><SlidersHorizontal size={15}/>Параметры{(sourceId || dateFrom || dateTo) && <span className="filter-dot"/>}<ChevronDown size={13}/></button><div className="composer-right"><span>Enter ↵</span><button className="send" type="submit" aria-label="Отправить вопрос" disabled={busy || !query.trim()}>{busy ? <LoaderCircle size={18} className="spin"/> : <ArrowUp size={20}/>}</button></div></div></form><div className="composer-note"><ShieldCheck size={12}/>Ответы основаны на сохранённых материалах. Проверяйте первоисточники.</div></div>
        </main>
        <aside className="evidence" id="sources-panel"><div className="evidence-heading"><div><BookOpen size={17}/><h2>Источники ответа</h2></div><span>{selectedSources.length.toString().padStart(2, '0')}</span></div><p className="evidence-intro">Материалы, на которые можно опереться.</p>{selectedSources.length ? <div className="evidence-list">{selectedSources.map((hit, i) => <article key={hit.id} className="evidence-card"><div className="evidence-card-top"><span className="reference-index">{i + 1}</span><span>{hit.source_name}</span><a href={sourceLink(hit.url)} target="_blank" rel="noreferrer" aria-label={`Открыть источник ${i + 1}`}><ArrowUpRight size={16}/></a></div><a className="evidence-title" href={sourceLink(hit.url)} target="_blank" rel="noreferrer">{hit.title}</a><div className="evidence-date">{when(hit.published_at)}</div><p className={expanded === hit.id ? 'excerpt expanded' : 'excerpt'}>{hit.text}</p><button className="read-fragment" onClick={() => setExpanded(expanded === hit.id ? null : hit.id)}>{expanded === hit.id ? 'Свернуть' : 'Читать фрагмент'}<ChevronDown size={13}/></button>{hit.aliases.length > 1 && <details className="aliases"><summary>Другие ссылки на материал</summary>{hit.aliases.filter(a => a.url !== hit.url).map(a => <a key={a.url} href={sourceLink(a.url)} target="_blank" rel="noreferrer">{a.title}</a>)}</details>}</article>)}</div> : <div className="evidence-empty"><div className="paper-illustration"><div/><div/><div/><BookOpen size={24}/></div><h3>Здесь будет основа ответа</h3><p>После вашего вопроса мы покажем<br/>найденные статьи и точные фрагменты.</p></div>}<div className="evidence-bottom"><span className="green-square"/>ПРОЗРАЧНЫЙ ПОИСК<span>↗</span></div></aside>
      </div> : !admin ? <main className="admin-main"><div className="section-kicker">ДОСТУП АДМИНИСТРАТОРА</div><h1>{view === 'lab' ? 'Исследовать поиск' : 'Управлять знаниями'}</h1><p className="muted">Войдите, чтобы добавлять материалы и сравнивать режимы поиска.</p><form className="login-card" onSubmit={login}><LockKeyhole size={25}/><h2>Вход в рабочее пространство</h2><label>Пароль администратора<input type="password" autoComplete="current-password" minLength={16} value={password} onChange={e => setPassword(e.target.value)} required/></label><button className="primary" disabled={adminBusy}>{adminBusy ? <LoaderCircle className="spin" size={16}/> : null}Войти<ArrowUpRight size={16}/></button></form></main> : view === 'admin' ? <main className="admin-main"><div className="page-heading"><div><div className="section-kicker">КОНТРОЛЬ НАД КОНТЕКСТОМ</div><h1>База знаний</h1><p className="muted">Добавляйте источники. Обновляйте тогда, когда нужно.</p></div><button className="secondary" onClick={() => action(async () => { await api('/admin/logout', { method: 'POST' }); csrf = ''; setAdmin(false); }, '')}><LogOut size={15}/>Выйти</button></div>{notice && <div role="status" className="alert success"><Check size={16}/>{notice}</div>}<div className="stats-grid"><div><Database size={18}/><strong>{status?.documents ?? '—'}</strong><span>Документов в базе</span></div><div><Layers3 size={18}/><strong>{status?.index?.chunks ?? '—'}</strong><span>Поисковых фрагментов</span></div><div><Sparkles size={18}/><strong>{status?.llm.configured ? 'Подключена' : 'Выдержки'}</strong><span>{status?.llm.configured ? status.llm.model : 'Ответы без внешней LLM'}</span></div></div><section className="admin-section"><div className="section-heading"><h2>Источники <span>{adminSources.length}</span></h2><div className="button-row"><button className="secondary" disabled={adminBusy} onClick={() => fileInput.current?.click()}><Upload size={15}/>Загрузить файл</button><input ref={fileInput} type="file" accept=".txt,.md,.markdown" className="visually-hidden" onChange={e => { const file = e.target.files?.[0]; if (file) action(async () => { const data = new FormData(); data.append('file', file); await api('/admin/upload', { method: 'POST', body: data }); }, 'Файл загружен. Индексирование поставлено в очередь.'); e.target.value = ''; }}/><button className="primary" disabled={adminBusy} onClick={() => action(() => api('/admin/jobs', { method: 'POST', body: JSON.stringify({ kind: 'ingest' }) }), 'Сбор поставлен в очередь.')}><RefreshCw size={15}/>Обновить все</button></div></div><div className="source-table">{adminSources.map(s => <div className="source-row" key={s.id}><div className="table-source-icon">{s.kind === 'upload' ? <FileText size={18}/> : <Globe2 size={18}/>}</div><div className="source-description"><strong>{s.name}</strong><span>{s.url}</span>{s.last_error && <em>{s.last_error}</em>}</div><span className="kind-tag">{s.kind.toUpperCase()}</span><div className="source-time">{s.last_run ? when(s.last_run) : 'Ещё не обновлялся'}</div><button className={s.enabled ? 'toggle on' : 'toggle'} aria-label={`${s.enabled ? 'Отключить' : 'Включить'} ${s.name}`} aria-pressed={!!s.enabled} disabled={adminBusy} onClick={() => action(() => api('/admin/sources/' + s.id, { method: 'PATCH', body: JSON.stringify({ enabled: !s.enabled }) }), 'Источник обновлён.')}><span/></button>{s.kind !== 'upload' && <button className="icon-button" aria-label={`Обновить ${s.name}`} disabled={adminBusy || !s.enabled} onClick={() => action(() => api('/admin/jobs', { method: 'POST', body: JSON.stringify({ source_id: s.id }) }), 'Сбор источника поставлен в очередь.')}><RefreshCw size={15}/></button>}</div>)}</div><form className="add-source" onSubmit={e => { e.preventDefault(); action(async () => { await api('/admin/sources', { method: 'POST', body: JSON.stringify({ name: sourceName, url: sourceUrl, kind: sourceKind }) }); setSourceName(''); setSourceUrl(''); }, 'Источник добавлен. Нажмите обновление для сбора.'); }}><input aria-label="Название источника" placeholder="Название источника" maxLength={120} value={sourceName} onChange={e => setSourceName(e.target.value)} required/><input aria-label="URL источника" type="url" placeholder="https://example.com/rss" value={sourceUrl} onChange={e => setSourceUrl(e.target.value)} required/><select aria-label="Тип источника" value={sourceKind} onChange={e => setSourceKind(e.target.value)}><option value="rss">RSS</option><option value="html">HTML</option></select><button className="secondary" disabled={adminBusy}><Plus size={16}/>Добавить</button></form></section><section className="admin-section"><div className="section-heading"><h2>Обработка материалов</h2><button className="text-button" disabled={adminBusy} onClick={() => action(() => api('/admin/jobs', { method: 'POST', body: JSON.stringify({ kind: 'index' }) }), 'Перестроение индекса поставлено в очередь.')}><RefreshCw size={14}/>Перестроить индекс</button></div>{jobs.length ? jobs.map(job => <JobCard key={job.id} job={job}/>) : <div className="empty-small">Задач пока нет. Обновите источник, чтобы наполнить базу.</div>}</section><section className="system-status"><ShieldCheck size={18}/><div><strong>{status?.index?.dense ? 'Гибридный индекс готов' : 'Dense ещё не готов; доступен BM25 при наличии индекса'}</strong><p>Обновления запускаются вручную. Ключи находятся на сервере.</p>{status?.index && status.index.revision !== status.revision && <p className="job-error">Индекс отстаёт от данных. Запустите перестроение.</p>}{status?.index?.warnings.map(w => <p key={w}>{w}</p>)}{Object.values(status?.models.errors || {}).map(w => <p key={w}>{w}</p>)}</div></section></main> : <main className="admin-main lab-main"><div className="section-kicker">ПРОВЕРЯЕМ ГИПОТЕЗЫ</div><h1>Лаборатория поиска</h1><p className="muted">Один вопрос. Три режима. Качество оценивается по найденным материалам, время включает решение Laya.</p><form onSubmit={compare} className="lab-form"><input aria-label="Вопрос для сравнения" placeholder="Введите вопрос для сравнения режимов" value={labQuery} maxLength={1000} onChange={e => setLabQuery(e.target.value)} required/><button className="primary" disabled={adminBusy}>{adminBusy ? <LoaderCircle className="spin" size={16}/> : <FlaskConical size={16}/>}Сравнить</button></form>{compared && <div className="comparison">{Object.entries(compared).map(([mode, result]) => <section className="compare-column" key={mode}><span className="section-kicker">{mode.toUpperCase()}</span><h2>{{ never: 'Исходный порядок', always: 'С rerank', auto: 'Решение Laya' }[mode]}</h2><div className="compare-time">{(result.meta.timings_ms.total / 1000).toFixed(2)}<span>сек.</span></div><div className="compare-meta">Rerank: {result.meta.rerank_applied ? 'выполнен' : 'пропущен'}<br/>Laya: {result.meta.gate.decision}{result.meta.gate.confidence !== null ? ` · ${result.meta.gate.confidence.toFixed(3)}` : ''}</div>{result.meta.warnings.map(w => <p className="compare-warning" key={w}>{w}</p>)}{result.hits.map((hit, i) => <a key={hit.id} href={sourceLink(hit.url)} target="_blank" rel="noreferrer" className="compare-hit"><span>{i + 1}</span><div><strong>{hit.title}</strong><p>{hit.text.slice(0, 160)}…</p></div></a>)}<details className="warnings"><summary>Длительности этапов</summary><pre>{JSON.stringify(result.meta.timings_ms, null, 2)}</pre></details></section>)}</div>}<div className="lab-note"><FlaskConical size={18}/><p>Уверенность Laya — сигнал модели. Польза rerank и экономия времени требуют проверки на размеченных вопросах. Первый запуск включает загрузку моделей.</p></div></main>}
    </div>
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
