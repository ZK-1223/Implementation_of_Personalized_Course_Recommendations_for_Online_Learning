/**
 * 通用 UI 工具：安全转义、格式化、卡片/KPI/徽章/环形分等模板片段。
 *
 * 只输出字符串模板（配合 innerHTML 使用），所有来自后端的文本必须经过 esc()，
 * 因此本文件是全站唯一的 XSS 防线。
 */

/** HTML 转义 */
export const esc = (value) =>
  String(value ?? '').replace(
    /[&<>"']/g,
    (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]
  );

/** 数字（保留 d 位小数） */
export const fx = (value, d = 2) => Number(value ?? 0).toFixed(d);

/** 概率 → 百分比字符串 */
export const pct = (value, d = 0) => `${(Number(value ?? 0) * 100).toFixed(d)}%`;

/** 小时数 */
export const hrs = (value) => `${Number(value ?? 0).toFixed(1)}<small>h</small>`;

/** 夹取 */
export const clamp01 = (value) => Math.min(1, Math.max(0, Number(value ?? 0)));

/** 难度点阵 */
export const levelDots = (level, max = 5) =>
  `<span class="level-dots" title="难度 ${level}/${max}">${Array.from(
    { length: max },
    (_, i) => `<i class="${i < level ? 'on' : ''}"></i>`
  ).join('')}</span>`;

/** 徽章 */
export const badge = (text, tone = '') =>
  `<span class="badge ${tone ? `badge-${tone}` : ''}">${esc(text)}</span>`;

/** 标签 */
export const chip = (text, variant = '') =>
  `<span class="chip ${variant}">${esc(text)}</span>`;

/** 进度条（宽度在挂载后由 animateBars 动画到位） */
export const bar = (value, tone = '', extra = '') =>
  `<div class="bar ${extra}"><div class="bar-fill ${tone}" data-w="${Math.round(clamp01(value) * 100)}"></div></div>`;

/** 度量行：标签 + 条 + 数值 */
export const meterRow = (label, value, { tone = '', display, title } = {}) => `
  <div class="meter">
    <span class="meter-label" title="${esc(title || label)}">${esc(label)}</span>
    ${bar(value, tone)}
    <span class="meter-value">${display ?? pct(value)}</span>
  </div>`;

/**
 * 环形分数（SVG）。
 * @param {number} value 0~1
 * @param {{size?:number, stroke?:number, color?:string, label?:string, text?:string}} opts
 */
export const ring = (value, opts = {}) => {
  const size = opts.size ?? 54;
  const stroke = opts.stroke ?? 5;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const ratio = clamp01(value);
  const offset = circumference * (1 - ratio);
  const color = opts.color ?? 'var(--brand)';
  return `
    <div class="ring" style="width:${size}px;height:${size}px">
      <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
        <circle class="ring-track" cx="${size / 2}" cy="${size / 2}" r="${radius}" stroke-width="${stroke}"></circle>
        <circle class="ring-progress" cx="${size / 2}" cy="${size / 2}" r="${radius}" stroke-width="${stroke}"
                stroke="${color}" stroke-dasharray="${circumference.toFixed(2)}"
                data-offset="${offset.toFixed(2)}" stroke-dashoffset="${circumference.toFixed(2)}"
                transform="rotate(0 ${size / 2} ${size / 2})"></circle>
      </svg>
      <div class="ring-center">
        <strong style="color:${color}">${opts.text ?? pct(ratio)}</strong>
        ${opts.label ? `<span>${esc(opts.label)}</span>` : ''}
      </div>
    </div>`;
};

/** KPI 卡片 */
export const kpi = ({ label, value, unit = '', foot = '', tone = '', icon = '' }) => `
  <div class="kpi ${tone}">
    <div class="kpi-label">${icon}${esc(label)}</div>
    <div class="kpi-value" data-count>${value}${unit ? `<small>${esc(unit)}</small>` : ''}</div>
    ${foot ? `<div class="kpi-foot">${foot}</div>` : ''}
  </div>`;

/** 卡片容器 */
export const card = ({ title = '', hint = '', actions = '', body = '', cls = '', bodyCls = '' }) => `
  <section class="card ${cls}">
    ${
      title
        ? `<div class="card-head">
             <div>
               <h2><span class="dot"></span>${esc(title)}</h2>
               ${hint ? `<p class="card-hint">${esc(hint)}</p>` : ''}
             </div>
             ${actions ? `<div class="card-actions">${actions}</div>` : ''}
           </div>`
        : ''
    }
    <div class="card-body ${bodyCls}">${body}</div>
  </section>`;

/** 顶部离线提示条 */
export const offlineBanner = (reason) => `
  <div class="offline-banner">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2.6 20h18.8L12 3Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 9v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="17" r="1.1"/></svg>
    <span>当前展示的是<strong>契约示例数据</strong>（离线演示模式）。${reason ? esc(reason) : ''}</span>
  </div>`;

/** 状态面板（错误 / 空态） */
export const statePanel = ({ title = '暂无数据', message = '', tone = 'warn', actionHtml = '' } = {}) => `
  <div class="state-panel">
    <div class="state-icon" style="color:${tone === 'danger' ? 'var(--danger)' : 'var(--warn)'}">
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.7"/><path d="M12 8v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="16.4" r="1.1"/></svg>
    </div>
    <h4>${esc(title)}</h4>
    ${message ? `<p>${message}</p>` : ''}
    ${actionHtml}
  </div>`;

/** 骨架屏（整页） */
export const skeletonView = (blocks = 3, hint = '') => {
  const cols = blocks === 'profile' ? ['col-8', 'col-4', 'col-6', 'col-6'] : null;
  const cells = cols
    ? cols.map((c) => `<div class="skeleton sk-card ${c}"></div>`).join('')
    : Array.from({ length: 3 }, () => '<div class="skeleton sk-card col-4"></div>').join('');
  return `
    <div class="view">
      ${hint ? `<p class="tiny muted" style="margin:-4px 0 2px 2px">${esc(hint)}</p>` : ''}
      <div class="skeleton-grid">
        ${cells}
      </div>
      <div class="skeleton-grid">
        <div class="skeleton sk-card col-7" style="min-height:260px"></div>
        <div class="skeleton sk-card col-5" style="min-height:260px"></div>
      </div>
    </div>`;
};

/** 轻提示 */
export const toast = (message, tone = '') => {
  const stack = document.getElementById('toast-stack');
  if (!stack) return;
  const node = document.createElement('div');
  node.className = `toast ${tone}`;
  node.textContent = message;
  stack.appendChild(node);
  setTimeout(() => {
    node.classList.add('hide');
    setTimeout(() => node.remove(), 300);
  }, 3600);
};

/** 是否处于「减少动态效果」偏好（同时用于截图稳定性） */
const reducedMotion = () =>
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** 挂载后统一触发「进度条生长」动画 */
export const animateBars = (root) => {
  const bars = root.querySelectorAll('.bar-fill[data-w]');
  if (reducedMotion()) {
    bars.forEach((node) => {
      node.style.width = `${node.getAttribute('data-w')}%`;
    });
    return;
  }
  requestAnimationFrame(() => {
    bars.forEach((node, index) => {
      const width = node.getAttribute('data-w');
      node.style.transitionDelay = `${Math.min(index * 22, 320)}ms`;
      node.style.width = `${width}%`;
    });
  });
};

/** 挂载后统一触发「环形分绘制」动画 */
export const animateRings = (root) => {
  const rings = root.querySelectorAll('.ring-progress[data-offset]');
  if (reducedMotion()) {
    rings.forEach((node) => {
      node.style.strokeDashoffset = node.getAttribute('data-offset');
    });
    return;
  }
  requestAnimationFrame(() => {
    rings.forEach((node, index) => {
      node.style.transitionDelay = `${Math.min(index * 60, 400)}ms`;
      node.style.strokeDashoffset = node.getAttribute('data-offset');
    });
  });
};

/** 数字滚动动画（KPI 用，仅对纯数字生效） */
export const animateCounts = (root) => {
  if (reducedMotion()) return;
  root.querySelectorAll('.kpi-value[data-count]').forEach((node) => {
    const raw = node.firstChild?.textContent?.trim() ?? '';
    const target = Number(raw.replace(/[^0-9.\-]/g, ''));
    if (!Number.isFinite(target) || Math.abs(target) < 0.001) return;
    const decimals = (raw.split('.')[1] || '').length;
    const unitNode = node.querySelector('small');
    const start = performance.now();
    const duration = 720;
    const tick = (now) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      const value = (target * eased).toFixed(decimals);
      node.firstChild.textContent = value;
      if (p < 1) requestAnimationFrame(tick);
      else node.firstChild.textContent = raw;
    };
    if (unitNode) node.insertBefore(document.createTextNode(''), unitNode);
    requestAnimationFrame(tick);
  });
};

/** 一次性激活所有进入动画 */
export const animateAll = (root) => {
  animateBars(root);
  animateRings(root);
  animateCounts(root);
};

/** 头像首字 */
export const initials = (name) => esc(String(name ?? '?').slice(0, 2));
