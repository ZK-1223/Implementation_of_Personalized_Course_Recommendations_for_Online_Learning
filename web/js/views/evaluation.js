/**
 * 视图 6：系统评测
 * 推荐算法离线指标对比、效果预测模型对比、消融实验、全量知识图谱。
 */

import { api } from '../core/api.js';
import { chartById, PALETTE, fmt3 } from '../core/charts.js';
import { MOCK } from '../mock.js';
import { bar, badge, card, chip, esc, fx, kpi, pct } from '../core/ui.js';

const METRICS = [
  ['precision_at_k', 'Precision@K', 3],
  ['recall_at_k', 'Recall@K', 3],
  ['ndcg_at_k', 'NDCG@K', 3],
  ['map', 'MAP', 3],
  ['coverage', 'Coverage', 3],
  ['diversity', 'Diversity', 3],
  ['novelty', 'Novelty', 3],
  ['hit_rate', 'HitRate', 3],
];

const FEATURE_LABELS = {
  learner_ability: '学习者能力',
  prereq_readiness: '先修就绪度',
  readiness: '课程就绪度',
  course_level_gap: '课程难度差',
  level_gap: '难度差',
  course_level: '课程难度等级',
  duration_hours: '课程时长',
  time_budget_ratio: '时间预算比',
  weekly_budget_log: '周预算（log）',
  style_fit: '风格匹配度',
  knowledge_gain_potential: '掌握度增益潜力',
  peer_completion: '同伴完成率',
  hist_completion_rate: '历史完成率',
  hist_avg_quiz: '历史测验均分',
  hist_course_count: '历史学习门数',
  hist_log_count: '历史学习频次（log）',
  course_rating: '课程评分',
  rating: '课程评分',
  enroll_count: '选课人数',
  enroll_log: '选课热度（log）',
  theory_ratio: '理论占比',
  motivation: '学习动机编码',
  domain_match: '领域匹配度',
  avg_mastery: '平均掌握度',
  target_avg_mastery: '目标点平均掌握度',
  gap_ratio: '缺口比例',
};

/** 未登记的特征名做一次可读化处理（snake_case → 空格分隔） */
const labelOf = (name) => FEATURE_LABELS[name] || String(name).replace(/_/g, ' ');

export default {
  id: 'evaluation',
  title: '系统评测',
  subtitle: '离线指标 · 推荐算法对比 · 消融实验 · 全量知识图谱',
  slowHint: '首次进入会执行离线评测：训练 4 个推荐器并做留一法评估，约需 30~60 秒（之后命中服务端缓存）',

  async load(ctx) {
    const [evaluation, graph] = await Promise.all([
      api.evaluation(),
      api.graph({ domain: ctx.params.domain || '' }),
    ]);
    return { evaluation, graph };
  },

  mock(ctx) {
    return { evaluation: MOCK.evaluation(), graph: MOCK.graph({ domain: ctx.params.domain || null }) };
  },

  render({ evaluation, graph }, ctx) {
    const rows = evaluation.recommenders || [];
    const best = {};
    METRICS.forEach(([key]) => {
      best[key] = Math.max(...rows.map((row) => Number(row[key] ?? 0)), 0);
    });

    const recommenderTable = `
      <div class="table-wrap">
        <table class="table">
          <thead>
            <tr>
              <th>推荐算法</th>
              ${METRICS.map(([, label]) => `<th class="num">${label}</th>`).join('')}
            </tr>
          </thead>
          <tbody>
            ${rows
              .map(
                (row) => `
              <tr class="${/Hybrid|本系统/i.test(row.name) ? 'row-highlight' : ''}">
                <td class="cell-strong">${esc(row.name)}</td>
                ${METRICS.map(([key, , digits]) => {
                  const value = Number(row[key] ?? 0);
                  const isBest = value >= best[key] && value > 0;
                  return `<td class="num ${isBest ? 'cell-best' : ''}">${fx(value, digits)}</td>`;
                }).join('')}
              </tr>`
              )
              .join('')}
          </tbody>
        </table>
      </div>`;

    const effect = evaluation.effect_model;
    const gain = evaluation.gain_model;
    const features = (effect?.features || []).slice().sort((a, b) => b.importance - a.importance);
    const ablation = evaluation.ablation || [];

    const effectTable = effect
      ? card({
          cls: 'col-6',
          title: '完成概率预测模型对比',
          hint: `任务 ${effect.task} · 正样本率 ${pct(effect.positive_rate)}`,
          actions: effect.best ? badge(`最优 ${effect.best}`, 'accent') : '',
          bodyCls: 'flush',
          body: `
            <div class="table-wrap">
              <table class="table">
                <thead>
                  <tr><th>模型</th><th class="num">AUC</th><th class="num">Accuracy</th><th class="num">F1</th>
                      <th class="num">CV 均值</th><th class="num">CV 标准差</th></tr>
                </thead>
                <tbody>
                  ${(effect.models || [])
                    .map(
                      (model) => `
                    <tr class="${model.name === effect.best ? 'row-highlight' : ''}">
                      <td class="cell-strong">${esc(model.name)}</td>
                      <td class="num">${fx(model.auc, 3)}</td>
                      <td class="num">${fx(model.accuracy, 3)}</td>
                      <td class="num">${fx(model.f1, 3)}</td>
                      <td class="num">${fx(model.cv_mean, 3)}</td>
                      <td class="num cell-muted">±${fx(model.cv_std, 3)}</td>
                    </tr>`
                    )
                    .join('')}
                </tbody>
              </table>
            </div>`,
        })
      : card({ cls: 'col-6', title: '完成概率预测模型对比', body: '<p class="muted tiny">评测模块未提供该段数据</p>' });

    const gainTable = gain
      ? card({
          cls: 'col-6',
          title: '掌握度增益回归模型对比',
          hint: `任务 ${gain.task}`,
          actions: gain.best ? badge(`最优 ${gain.best}`, 'accent') : '',
          bodyCls: 'flush',
          body: `
            <div class="table-wrap">
              <table class="table">
                <thead><tr><th>模型</th><th class="num">MAE</th><th class="num">RMSE</th><th class="num">R²</th></tr></thead>
                <tbody>
                  ${(gain.models || [])
                    .map(
                      (model) => `
                    <tr class="${model.name === gain.best ? 'row-highlight' : ''}">
                      <td class="cell-strong">${esc(model.name)}</td>
                      <td class="num">${fx(model.mae, 3)}</td>
                      <td class="num">${fx(model.rmse, 3)}</td>
                      <td class="num">${fx(model.r2, 3)}</td>
                    </tr>`
                    )
                    .join('')}
                </tbody>
              </table>
            </div>`,
        })
      : card({ cls: 'col-6', title: '掌握度增益回归模型对比', body: '<p class="muted tiny">评测模块未提供该段数据</p>' });

    const ablationTable = `
      <div class="table-wrap">
        <table class="table">
          <thead><tr><th>消融变体</th><th>NDCG@K</th><th class="num">Δ NDCG</th></tr></thead>
          <tbody>
            ${ablation
              .map((row) => {
                const delta = Number(row.delta ?? 0);
                const color = delta < 0 ? 'var(--danger)' : delta > 0 ? 'var(--accent)' : 'var(--text-3)';
                return `
              <tr>
                <td class="cell-strong">${esc(row.variant)}</td>
                <td><div class="cell-bar">${bar(row.ndcg_at_k ?? 0, delta < 0 ? 'warn' : 'accent')}<span>${fx(
                  row.ndcg_at_k,
                  3
                )}</span></div></td>
                <td class="num" style="color:${color}">${delta === 0 ? '基准' : `${delta > 0 ? '+' : ''}${fx(delta, 3)}`}</td>
              </tr>`;
              })
              .join('')}
          </tbody>
        </table>
      </div>`;

    const top = rows.reduce((acc, row) => (Number(row.ndcg_at_k) > Number(acc?.ndcg_at_k ?? -1) ? row : acc), null);
    const kpiRow = `
      <div class="kpi-grid">
        ${kpi({ label: '参与对比算法', value: rows.length, unit: '个', tone: '', foot: 'Popularity / Content / CF / Hybrid' })}
        ${kpi({
          label: '最优 NDCG@K',
          value: fx(top?.ndcg_at_k ?? 0, 3),
          tone: 'accent',
          foot: top ? esc(top.name) : '',
        })}
        ${kpi({
          label: '最优覆盖率',
          value: pct(Math.max(...rows.map((row) => Number(row.coverage ?? 0)), 0)),
          tone: 'cyan',
          foot: '长尾课程挖掘能力',
        })}
        ${kpi({
          label: '消融变体',
          value: ablation.length,
          unit: '组',
          tone: 'violet',
          foot: '逐项验证模块贡献',
        })}
        ${kpi({
          label: '效果模型 AUC',
          value: fx(Math.max(...(effect?.models || []).map((m) => Number(m.auc ?? 0)), 0), 3),
          tone: 'warn',
          foot: effect?.best ? `最优 ${esc(effect.best)}` : '无模型数据',
        })}
      </div>`;

    const domainOptions = (ctx.meta?.domains || [])
      .map(
        (domain) =>
          `<option value="${esc(domain.name)}" ${ctx.params.domain === domain.name ? 'selected' : ''}>${esc(
            domain.name
          )}</option>`
      )
      .join('');

    return `
      ${kpiRow}
      <div class="grid">
        ${card({
          cls: 'col-12',
          title: '推荐算法离线指标对比',
          hint: '留一法划分 · 每位学习者保留 20% 正反馈作为测试集',
          actions: `<span class="tiny muted">生成时间 ${esc(String(evaluation.generated_at || '').replace('T', ' '))}</span>`,
          bodyCls: 'flush',
          body: recommenderTable,
        })}
      </div>
      <div class="grid">
        ${card({
          cls: 'col-7',
          title: '关键指标对比',
          hint: '四条主指标横向对比，越高越好',
          body: `<div class="chart chart-md" id="eval-metrics"></div>`,
        })}
        ${card({
          cls: 'col-5',
          title: '特征重要性',
          hint: '完成概率预测模型中各特征的相对贡献',
          body: `<div class="chart chart-md" id="eval-features"></div>`,
        })}
      </div>
      <div class="grid">${effectTable}${gainTable}</div>
      <div class="grid">
        ${card({
          cls: 'col-7',
          title: '消融实验',
          hint: '逐项移除系统模块后 NDCG@K 的变化；Δ>0 表示移除该模块后指标反而上升（说明该模块牺牲了精度换取其他性质）',
          bodyCls: 'flush',
          body: ablationTable,
        })}
        ${card({
          cls: 'col-5',
          title: '评测口径说明',
          hint: '指标定义与实验设置',
          body: `<ul class="notes-list">${(evaluation.notes || [])
            .map((note) => `<li>${esc(note)}</li>`)
            .join('')}</ul>`,
        })}
      </div>
      <div class="grid">
        ${card({
          cls: 'col-12',
          title: '全量知识图谱',
          hint: '支持领域筛选、拖拽与缩放；节点大小随难度增长',
          actions: `
            <select class="mini-select" id="eval-domain">
              <option value="">全部领域</option>
              ${domainOptions}
            </select>
            ${chip(`${(graph.nodes || []).length} 节点 / ${(graph.links || []).length} 边`, 'tiny ghost')}`,
          body: `<div class="chart chart-xl" id="eval-graph"></div>`,
        })}
      </div>`;
  },

  mount({ evaluation, graph }, ctx) {
    /* ---------------- 关键指标对比 ---------------- */
    const rows = evaluation.recommenders || [];
    const shortName = (name) =>
      name.includes('Popularity')
        ? '热度基线'
        : name.includes('Content')
          ? '内容推荐'
          : name.includes('CF')
            ? 'CF-SVD'
            : '混合（本系统）';
    const keys = ['precision_at_k', 'recall_at_k', 'ndcg_at_k', 'map'];
    chartById('eval-metrics', {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0, data: ['Precision@K', 'Recall@K', 'NDCG@K', 'MAP'] },
      grid: { left: 10, right: 16, top: 42, bottom: 6, containLabel: true },
      xAxis: { type: 'category', data: rows.map((row) => shortName(row.name)) },
      yAxis: { type: 'value', axisLabel: { formatter: (v) => v.toFixed(2) } },
      series: keys.map((key, index) => ({
        name: { precision_at_k: 'Precision@K', recall_at_k: 'Recall@K', ndcg_at_k: 'NDCG@K', map: 'MAP' }[key],
        type: 'bar',
        data: rows.map((row) => row[key]),
        itemStyle: { color: PALETTE[index], borderRadius: [4, 4, 0, 0] },
        barMaxWidth: 14,
      })),
    });

    /* ---------------- 特征重要性 ---------------- */
    const features = ((evaluation.effect_model?.features) || [])
      .slice()
      .sort((a, b) => a.importance - b.importance);
    chartById('eval-features', {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v) => fx(v, 3) },
      grid: { left: 10, right: 30, top: 12, bottom: 6, containLabel: true },
      xAxis: { type: 'value', axisLabel: { formatter: (v) => v.toFixed(2) } },
      yAxis: {
        type: 'category',
        data: features.map((item) => labelOf(item.name)),
        axisLabel: { fontSize: 10.5 },
      },
      series: [
        {
          type: 'bar',
          data: features.map((item, index) => ({
            value: item.importance,
            itemStyle: {
              color: index === features.length - 1 ? PALETTE[1] : PALETTE[0],
              borderRadius: [0, 4, 4, 0],
            },
          })),
          barMaxWidth: 14,
          label: { show: true, position: 'right', color: '#8494b5', fontSize: 10, formatter: (p) => fmt3(p.value) },
        },
      ],
    });

    /* ---------------- 全量知识图谱 ---------------- */
    const categories = (graph.categories || []).map((name) => ({ name }));
    const manyNodes = (graph.nodes || []).length > 120;
    chartById('eval-graph', {
      tooltip: {
        formatter: (params) =>
          params.dataType === 'edge'
            ? `${params.data.source} → ${params.data.target}`
            : `<b>${params.data.name}</b><br/>${params.data.id}<br/>难度 ${params.data.value ?? '-'}`,
      },
      legend: [{ data: categories.map((c) => c.name), bottom: 0, type: 'scroll' }],
      series: [
        {
          type: 'graph',
          layout: 'force',
          roam: true,
          draggable: true,
          categories,
          data: (graph.nodes || []).map((node) => ({
            id: node.id,
            name: node.name,
            category: node.category,
            symbolSize: Math.max(8, node.symbolSize * (manyNodes ? 0.72 : 1)),
            value: node.difficulty,
            itemStyle: {
              color: PALETTE[node.category % PALETTE.length],
              borderColor: 'rgba(9,14,26,0.85)',
              borderWidth: 1,
            },
            // 节点过多时默认不显示文字标签，悬停/聚焦时再显示，避免糊成一团
            label: { show: !manyNodes && node.depth <= 0, fontSize: 10, color: '#cfdcf7' },
          })),
          links: (graph.links || []).map((link) => ({
            source: link.source,
            target: link.target,
            lineStyle: { width: 1, opacity: 0.35, curveness: 0.1 },
          })),
          force: { repulsion: 130, edgeLength: [22, 56], gravity: 0.08 },
          emphasis: {
            focus: 'adjacency',
            label: { show: true, fontSize: 11, color: '#ffffff', fontWeight: 600 },
            lineStyle: { width: 2, opacity: 0.95 },
          },
          labelLayout: { hideOverlap: true },
          scaleLimit: { min: 0.4, max: 4 },
        },
      ],
    });

    /* ---------------- 领域筛选 ---------------- */
    const select = document.getElementById('eval-domain');
    select?.addEventListener('change', () => {
      ctx.setParams({ domain: select.value });
    });
  },
};
