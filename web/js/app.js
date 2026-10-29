/**
 * 单页应用入口：hash 路由、全局状态（学习者 / 方案预设）、视图生命周期。
 *
 * 视图契约：
 *   { id, title, subtitle, load(ctx), mock(ctx), render(data, ctx), mount?(data, ctx), bind?(root, ctx) }
 *
 * 数据获取策略：先调真实接口；若返回 503（算法模块尚未就绪）或后端不可达，
 * 自动回退到 mock.js 的契约示例数据，并在页面顶部展示「离线演示模式」提示条。
 */

import { api, describeError, isNotReady } from './core/api.js';
import { bindGlobalResize, disposeCharts } from './core/charts.js';
import { animateAll, esc, offlineBanner, skeletonView, statePanel, toast } from './core/ui.js';

import profile from './views/profile.js';
import recommend from './views/recommend.js';
import path from './views/path.js';
import dashboard from './views/dashboard.js';
import social from './views/social.js';
import evaluation from './views/evaluation.js';

const VIEWS = { profile, recommend, path, dashboard, social, evaluation };
const DEFAULT_VIEW = 'profile';
const LEARNER_KEY = 'learnpath.learner';
const PRESET_KEY = 'learnpath.preset';

const state = {
  meta: null,
  learnerId: null,
  preset: 'balanced',
  view: DEFAULT_VIEW,
  paramsByView: {},
  backend: 'loading', // loading | ok | offline
  offlineReason: '',
  version: '1.0.0',
};

const dom = {
  root: document.getElementById('view-root'),
  nav: document.getElementById('nav'),
  select: document.getElementById('learner-select'),
  preset: document.getElementById('preset-switch'),
  title: document.getElementById('view-title'),
  subtitle: document.getElementById('view-subtitle'),
  refresh: document.getElementById('refresh-btn'),
  sourceChip: document.getElementById('data-source-chip'),
  sourceText: document.getElementById('data-source-text'),
  version: document.getElementById('app-version'),
};

let renderToken = 0;

/* ============================================================
   小工具
   ============================================================ */

const store = {
  get(key) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      /* 隐私模式下忽略 */
    }
  },
};

function setSource(mode, text) {
  if (!dom.sourceChip) return;
  dom.sourceChip.classList.toggle('offline', mode === 'offline');
  dom.sourceText.textContent = text;
}

/* ============================================================
   外壳交互
   ============================================================ */

function fillLearnerSelect() {
  const learners = state.meta?.learners || [];
  // 720 名学习者按目标分组，避免下拉框变成一条长名单
  const groups = new Map();
  learners.forEach((learner) => {
    const key = learner.goal_name || learner.goal_id || '其他';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(learner);
  });

  dom.select.innerHTML = [...groups.entries()]
    .map(
      ([goalName, items]) => `
      <optgroup label="${esc(goalName)}（${items.length} 人）">
        ${items
          .map(
            (learner) =>
              `<option value="${esc(learner.learner_id)}" ${
                learner.learner_id === state.learnerId ? 'selected' : ''
              }>${esc(learner.nickname)} · 能力 ${Number(learner.ability ?? 0).toFixed(2)}</option>`
          )
          .join('')}
      </optgroup>`
    )
    .join('');
}

function syncPresetButtons() {
  dom.preset.querySelectorAll('button').forEach((button) => {
    const active = button.getAttribute('data-preset') === state.preset;
    button.classList.toggle('active', active);
    button.setAttribute('aria-selected', String(active));
  });
}

function updateNav() {
  dom.nav.querySelectorAll('.nav-item').forEach((item) => {
    item.classList.toggle('active', item.getAttribute('data-view') === state.view);
  });
}

function currentParams() {
  if (!state.paramsByView[state.view]) state.paramsByView[state.view] = {};
  return state.paramsByView[state.view];
}

function buildCtx() {
  const learner = (state.meta?.learners || []).find((item) => item.learner_id === state.learnerId);
  const params = currentParams();
  return {
    learnerId: state.learnerId,
    learner,
    preset: state.preset,
    params,
    meta: state.meta,
    backend: state.backend,
    setParams(patch) {
      Object.assign(params, patch);
      renderView();
    },
    setPreset(preset) {
      if (!preset || preset === state.preset) return;
      state.preset = preset;
      store.set(PRESET_KEY, preset);
      syncPresetButtons();
      renderView();
      toast(`已切换到${preset === 'fast' ? '速通' : preset === 'thorough' ? '夯实' : '均衡'}方案，路径与推荐已重算`);
    },
    reload: renderView,
  };
}

/* ============================================================
   视图渲染
   ============================================================ */

async function renderView() {
  const def = VIEWS[state.view] || VIEWS[DEFAULT_VIEW];
  const token = (renderToken += 1);

  disposeCharts();
  dom.title.textContent = def.title;
  dom.subtitle.textContent = def.subtitle;
  updateNav();
  dom.root.innerHTML = skeletonView('profile', def.slowHint || '');
  window.scrollTo({ top: 0, behavior: 'smooth' });

  const ctx = buildCtx();
  let data = null;
  let offline = state.backend === 'offline';
  let reason = state.offlineReason;

  try {
    data = await def.load(ctx);
  } catch (error) {
    if (isNotReady(error) || state.backend === 'offline') {
      data = def.mock(ctx);
      offline = true;
      reason = describeError(error);
      if (state.backend === 'ok') {
        toast(`「${def.title}」接口暂不可用，已用契约示例数据渲染`, 'warn');
      }
    } else {
      if (token !== renderToken) return;
      dom.root.innerHTML = `<div class="view">${statePanel({
        title: '数据加载失败',
        tone: 'danger',
        message: `${esc(describeError(error))}<br/>请确认后端已启动：<code>uvicorn app.main:app --port 8000</code>`,
        actionHtml: '<button class="btn ghost" id="retry-btn" type="button">重试</button>',
      })}</div>`;
      document.getElementById('retry-btn')?.addEventListener('click', renderView);
      return;
    }
  }

  if (token !== renderToken) return;

  const banner = offline ? offlineBanner(reason || '后端接口暂不可用。') : '';
  dom.root.innerHTML = `<div class="view">${banner}${def.render(data, ctx)}</div>`;
  animateAll(dom.root);

  try {
    def.mount?.(data, ctx);
  } catch (error) {
    console.error('[chart] mount failed', error);
  }
  try {
    def.bind?.(dom.root, ctx);
  } catch (error) {
    console.error('[view] bind failed', error);
  }
}

/* ============================================================
   路由
   ============================================================ */

function parseHash() {
  const key = (window.location.hash || '').replace(/^#\/?/, '').trim();
  return VIEWS[key] ? key : DEFAULT_VIEW;
}

function onRouteChange() {
  const next = parseHash();
  if (!window.location.hash || !VIEWS[next]) {
    window.history.replaceState(null, '', `#${next}`);
  }
  if (next === state.view && dom.root.querySelector('.view')) return;
  state.view = next;
  state.paramsByView[next] = state.paramsByView[next] || {};
  renderView();
}

/* ============================================================
   启动
   ============================================================ */

async function boot() {
  bindGlobalResize();

  /* ---- 1. 元数据 ---- */
  setSource('loading', '数据集加载中…');
  try {
    state.meta = await api.meta();
    state.backend = 'ok';
  } catch (error) {
    state.meta = (await import('./mock.js')).mockMeta();
    state.backend = 'offline';
    state.offlineReason = describeError(error);
    toast('后端未就绪，已切换为离线示例数据（界面与交互保持一致）', 'warn');
  }

  /* ---- 2. 版本号（可选） ---- */
  try {
    const health = await api.health();
    state.version = health.version || state.version;
    if (health.dataset_loaded === false) toast('后端数据层尚未就绪', 'warn');
  } catch {
    /* 离线时忽略 */
  }
  dom.version.textContent = `v${state.version}`;

  /* ---- 3. 学习者与预设 ---- */
  const learners = state.meta?.learners || [];
  const stored = store.get(LEARNER_KEY);
  state.learnerId = learners.some((item) => item.learner_id === stored)
    ? stored
    : learners[0]?.learner_id || '';
  const storedPreset = store.get(PRESET_KEY);
  if (storedPreset && ['balanced', 'fast', 'thorough'].includes(storedPreset)) {
    state.preset = storedPreset;
  }

  fillLearnerSelect();
  syncPresetButtons();

  const sourceLabel =
    state.backend === 'offline'
      ? '离线演示数据'
      : state.meta?.data_source === 'full'
        ? '完整数据集已加载'
        : '微型数据集（fixture）';
  setSource(state.backend === 'offline' ? 'offline' : 'ok', sourceLabel);

  /* ---- 4. 事件绑定 ---- */
  dom.select.addEventListener('change', () => {
    state.learnerId = dom.select.value;
    store.set(LEARNER_KEY, state.learnerId);
    state.paramsByView = {};
    renderView();
    const option = dom.select.options[dom.select.selectedIndex];
    toast(`已切换到 ${option ? option.textContent : state.learnerId}，各视图已刷新`);
  });

  dom.preset.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-preset]');
    if (button) buildCtx().setPreset(button.getAttribute('data-preset'));
  });

  dom.refresh.addEventListener('click', () => {
    dom.refresh.classList.add('spinning');
    setTimeout(() => dom.refresh.classList.remove('spinning'), 700);
    renderView();
    toast('已刷新当前视图数据');
  });

  window.addEventListener('hashchange', onRouteChange);

  /* ---- 5. 数字键快速切换视图 ---- */
  window.addEventListener('keydown', (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
    const keys = Object.keys(VIEWS);
    const index = Number(event.key) - 1;
    if (index >= 0 && index < keys.length) {
      window.location.hash = `#${keys[index]}`;
    }
  });

  /* ---- 6. 首屏 ---- */
  state.view = parseHash();
  if (!window.location.hash) window.history.replaceState(null, '', `#${state.view}`);
  renderView();
}

boot().catch((error) => {
  console.error(error);
  dom.root.innerHTML = `<div class="view">${statePanel({
    title: '应用初始化失败',
    tone: 'danger',
    message: esc(String(error?.message || error)),
    actionHtml: '<button class="btn ghost" onclick="location.reload()">重新加载</button>',
  })}</div>`;
});
