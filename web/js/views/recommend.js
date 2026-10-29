/**
 * 视图 2：智能推荐
 * 推荐卡片（综合分环形进度 + 5 项分项得分 + 中文可解释理由 + 预期增益 + 完成概率/风险）。
 */

import { api, describeError } from '../core/api.js';
import { chartById, PALETTE, fmtPct } from '../core/charts.js';
import { MOCK } from '../mock.js';
import { bar, badge, card, chip, esc, fx, kpi, levelDots, pct, ring } from '../core/ui.js';

const COMPONENTS = [
  ['content', '内容匹配'],
  ['collaborative', '协同过滤'],
  ['quality', '课程质量'],
  ['readiness', '就绪度'],
  ['social', '社交信号'],
];

const riskTone = (level) => (level === '低' ? 'accent' : level === '中' ? 'warn' : 'danger');

const scoreColor = (value) =>
  value >= 0.85 ? 'var(--accent)' : value >= 0.7 ? 'var(--brand)' : 'var(--warn)';

export default {
  id: 'recommend',
  title: '智能推荐',
  subtitle: '混合推荐 Top-K · 可解释理由 · 预期增益与完成概率',
  slowHint: '首次推荐会训练/加载学习效果预测模型，约需数秒…',

  async load(ctx) {
    return api.recommend(ctx.learnerId, ctx.params.k ?? 10);
  },

  mock(ctx) {
    return MOCK.recommend(ctx.learnerId, ctx.params.k ?? 10);
  },

  render(data, ctx) {
    const items = data.items || [];
    const preset = (ctx.meta?.presets || []).find((p) => p.key === ctx.preset);
    const weights = preset?.weights || { gain: 0.4, time: 0.25, smoothness: 0.2, fit: 0.15 };
    const k = ctx.params.k ?? 10;
    const avg = (key) => items.reduce((sum, item) => sum + (item[key] ?? 0), 0) / Math.max(1, items.length);
    const lowRisk = items.filter((item) => item.risk_level === '低').length;

    const toolbar = `
      <div class="toolbar">
        <div class="toolbar-group">
          <span class="toolbar-label">推荐条数 k</span>
          <input class="slider" id="k-slider" type="range" min="3" max="20" step="1" value="${k}"
                 style="--pct:${((k - 3) / 17) * 100}%" aria-label="推荐条数" />
          <span class="badge badge-brand" id="k-value">${k}</span>
          <span class="toolbar-label">当前方案：${esc(preset?.name || '均衡方案')}</span>
        </div>
        <div class="toolbar-group">
          <span class="toolbar-label">方案目标权重</span>
          ${chip(`收益 ${fx(weights.gain, 2)}`, 'tiny')}
          ${chip(`时间 ${fx(weights.time, 2)}`, 'tiny')}
          ${chip(`平滑 ${fx(weights.smoothness, 2)}`, 'tiny')}
          ${chip(`匹配 ${fx(weights.fit, 2)}`, 'tiny')}
        </div>
      </div>`;

    const kpis = `
      <div class="kpi-grid">
        ${kpi({ label: '平均综合分', value: fx(avg('score'), 3), tone: '', foot: `Top-${items.length} 课程` })}
        ${kpi({ label: '平均预期增益', value: fx(avg('expected_gain'), 3), tone: 'accent', foot: '掌握度提升均值' })}
        ${kpi({ label: '平均完成概率', value: pct(avg('p_completion')), tone: 'cyan', foot: '效果预测模型输出' })}
        ${kpi({ label: '低风险课程', value: lowRisk, unit: '门', tone: 'violet', foot: '完成概率 ≥ 0.70' })}
      </div>`;

    const recCards = items
      .map((item, index) => {
        const comps = COMPONENTS.map(
          ([key, label]) => `
            <div class="component-row">
              <span>${label}</span>
              ${bar(item.components?.[key] ?? 0, index === 0 ? 'accent' : '')}
              <b>${fx(item.components?.[key] ?? 0, 2)}</b>
            </div>`
        ).join('');
        return `
        <article class="rec-card ${index === 0 ? 'top-1' : ''}" style="animation-delay:${index * 55}ms">
          <div class="rec-top">
            <span class="rec-rank">${item.rank}</span>
            <div class="rec-title-block">
              <div class="rec-title">${esc(item.title)}</div>
              <div class="rec-subline">
                ${chip(item.domain, 'tiny')}
                ${badge(item.modality_name || item.modality, 'brand')}
                <span>${levelDots(item.level)}</span>
                <span>${esc(item.duration_hours)}h</span>
                <span>★ ${fx(item.rating, 1)}</span>
                <span>${Number(item.enroll_count).toLocaleString('zh-CN')} 人在学</span>
              </div>
            </div>
            <div class="rec-score">
              ${ring(item.score, {
                size: 58,
                stroke: 5,
                color: scoreColor(item.score),
                label: '综合分',
                text: fx(item.score, 3),
              })}
            </div>
          </div>

          <div class="rec-metrics">
            <div class="rec-metric"><span>预期增益</span><strong style="color:#5ee7bc">+${fx(
              item.expected_gain,
              3
            )}</strong></div>
            <div class="rec-metric"><span>完成概率</span><strong>${pct(item.p_completion)}</strong></div>
            <div class="rec-metric"><span>风险等级</span><strong>${badge(item.risk_level, riskTone(item.risk_level))}</strong></div>
            <div class="rec-metric"><span>时间可承受</span><strong>${pct(item.time_fit)}</strong></div>
          </div>

          <div class="component-bars">${comps}</div>

          <div class="reasons">
            ${(item.reasons || []).map((reason) => `<div class="reason">${esc(reason)}</div>`).join('')}
          </div>

          <div class="rec-foot">
            <div class="tag-list">
              ${(item.kp_names || []).slice(0, 4).map((name) => chip(name, 'tiny')).join('')}
            </div>
            <span class="tiny muted">${esc(item.course_id)} · 就绪度 ${fx(item.readiness, 2)} · 风格匹配 ${fx(
              item.style_fit,
              2
            )}</span>
          </div>
        </article>`;
      })
      .join('');

    const compareCard = card({
      cls: 'col-12',
      title: '分项得分对比',
      hint: 'Top-5 课程的五路信号强度（混合推荐融合前的归一化得分）',
      body: `<div class="chart chart-md" id="rec-compare"></div>`,
    });

    /* ---- 自适应反馈演示（要求 ⑥：反馈 → 掌握度更新 → 路径重规划） ---- */
    const feedbackCard = `
      <section class="card">
        <div class="card-head">
          <div>
            <h2><span class="dot"></span>自适应反馈演示</h2>
            <p class="card-hint">提交一次学习反馈，引擎将按 BKT 更新掌握度并自动重规划路径（真实调用 POST /api/feedback）</p>
          </div>
          <div class="card-actions">${chip('实现要求 ⑥', 'tiny violet')}</div>
        </div>
        <div class="card-body">
          <div class="toolbar-group" style="gap:14px">
            <label class="field-label">课程
              <select class="mini-select" id="fb-course">
                ${items
                  .map(
                    (item) =>
                      `<option value="${esc(item.course_id)}">${esc(item.title)}</option>`
                  )
                  .join('')}
              </select>
            </label>
            <label class="field-label">进度
              <input class="mini-input" id="fb-progress" type="number" min="0" max="1" step="0.05" value="1" />
            </label>
            <label class="field-label">测验分
              <input class="mini-input" id="fb-quiz" type="number" min="0" max="1" step="0.01" value="0.9" />
            </label>
            <label class="field-label">评分
              <input class="mini-input" id="fb-rating" type="number" min="1" max="5" step="0.5" value="4.5" />
            </label>
            <button class="btn" id="fb-submit" type="button">提交反馈并重规划</button>
          </div>
          <div id="fb-result"></div>
        </div>
      </section>`;

    return `
      ${toolbar}
      ${feedbackCard}
      ${kpis}
      <div class="rec-grid">${recCards || '<p class="muted">暂无推荐结果</p>'}</div>
      <div class="grid">${compareCard}</div>`;
  },

  mount(data) {
    const items = (data.items || []).slice(0, 5);
    chartById('rec-compare', {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: fmtPct },
      legend: { top: 0 },
      grid: { left: 10, right: 16, top: 42, bottom: 6, containLabel: true },
      xAxis: {
        type: 'category',
        data: items.map((item) => (item.title.length > 12 ? `${item.title.slice(0, 12)}…` : item.title)),
        axisLabel: { interval: 0, width: 90, overflow: 'truncate' },
      },
      yAxis: { type: 'value', max: 1, axisLabel: { formatter: (v) => `${Math.round(v * 100)}%` } },
      series: COMPONENTS.map(([key, label], index) => ({
        name: label,
        type: 'bar',
        data: items.map((item) => item.components?.[key] ?? 0),
        itemStyle: { color: PALETTE[index], borderRadius: [4, 4, 0, 0] },
        barMaxWidth: 16,
      })),
    });
  },

  bind(root, ctx) {
    const slider = root.querySelector('#k-slider');
    if (slider) {
      const label = root.querySelector('#k-value');
      let timer = null;
      slider.addEventListener('input', () => {
        const value = Number(slider.value);
        slider.style.setProperty('--pct', `${((value - 3) / 17) * 100}%`);
        if (label) label.textContent = value;
        clearTimeout(timer);
        timer = setTimeout(() => ctx.setParams({ k: value }), 320);
      });
    }

    /* ---- 自适应反馈提交 ---- */
    const submit = root.querySelector('#fb-submit');
    if (submit) {
      submit.addEventListener('click', async () => {
        const courseId = root.querySelector('#fb-course')?.value;
        const progress = Number(root.querySelector('#fb-progress')?.value ?? 1);
        const quiz = Number(root.querySelector('#fb-quiz')?.value ?? 0.9);
        const rating = Number(root.querySelector('#fb-rating')?.value ?? 4.5);
        const box = root.querySelector('#fb-result');
        if (!courseId || !box) return;

        submit.disabled = true;
        submit.textContent = '提交中…';
        try {
          const result = await api.feedback({
            learner_id: ctx.learnerId,
            course_id: courseId,
            progress: Math.min(1, Math.max(0, progress)),
            quiz_score: Math.min(1, Math.max(0, quiz)),
            rating: Math.min(5, Math.max(1, rating)),
            preset: ctx.preset,
          });
          const updated = Object.keys(result.mastery_updates || {});
          const deltas = Object.entries(result.delta || {})
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 3);
          box.innerHTML = `
            <div class="feedback-result">
              <h4>
                ${badge(result.replanned ? '路径已重规划' : '掌握度已更新', 'accent')}
                <span class="tiny muted">更新 ${updated.length} 个知识点 · 新路径 ${
                  result.path?.steps?.length ?? 0
                } 门课 / ${result.path?.total_hours ?? 0}h / ${result.path?.weeks ?? 0} 周</span>
              </h4>
              <div class="reason">${esc(result.reason || '')}</div>
              <div class="tag-list" style="margin-bottom:8px">
                ${(result.actions || []).map((action) => chip(action, 'tiny')).join('')}
                ${deltas
                  .map(([kp, delta]) => chip(`${kp} ${delta >= 0 ? '+' : ''}${fx(delta, 3)}`, 'tiny solid'))
                  .join('')}
              </div>
              <button class="btn ghost" id="fb-refresh" type="button">按最新掌握度刷新本页</button>
            </div>`;
          box.querySelector('#fb-refresh')?.addEventListener('click', () => ctx.reload());
          toast('反馈已生效，掌握度与路径已更新');
        } catch (error) {
          box.innerHTML = `<div class="feedback-result" style="background:rgba(255,91,110,.08);box-shadow:inset 0 0 0 1px rgba(255,91,110,.24)">
            <div class="reason" style="color:#ff9aa6">提交失败：${esc(describeError(error))}</div>
          </div>`;
        } finally {
          submit.disabled = false;
          submit.textContent = '提交反馈并重规划';
        }
      });
    }
  },
};
