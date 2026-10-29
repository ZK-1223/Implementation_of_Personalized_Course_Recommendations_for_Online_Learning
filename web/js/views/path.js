/**
 * 视图 3：学习路径
 * 三种预设方案对比、路径时间轴、帕累托前沿散点、周计划、知识图谱局部高亮。
 */

import { api } from '../core/api.js';
import { chartById, PALETTE, fmt3 } from '../core/charts.js';
import { MOCK } from '../mock.js';
import { bar, badge, card, chip, esc, fx, kpi, pct, ring } from '../core/ui.js';

const PRESET_LABEL = { balanced: '均衡方案', fast: '速通方案', thorough: '夯实方案' };

/** 掌握度 → 颜色（红 → 琥珀 → 绿） */
const masteryColor = (value) => {
  if (value == null) return '#26324d';
  const clamped = Math.min(1, Math.max(0, value));
  const hue = 10 + 150 * clamped;
  return `hsl(${hue.toFixed(0)} 70% 52%)`;
};

/**
 * 从全量图谱中截取「局部子图」：目标知识点 + 路径知识点 + 它们的一跳先修/后继。
 *
 * 全量图谱在完整数据集下有 400+ 节点，力导向布局既慢又糊，
 * 因此这里在客户端做一次聚焦裁剪（服务端仍返回全量，用于系统评测页展示）。
 */
function localSubgraph(graph, limit = 150) {
  const nodes = graph.nodes || [];
  if (nodes.length <= limit) return graph;

  const focus = new Set([...(graph.focus?.target_kps || []), ...(graph.focus?.path_kps || [])]);
  if (!focus.size) {
    const kept = nodes.slice(0, limit);
    const ids = new Set(kept.map((node) => node.id));
    return {
      ...graph,
      nodes: kept,
      links: (graph.links || []).filter((link) => ids.has(link.source) && ids.has(link.target)),
    };
  }

  const keep = new Set(focus);
  (graph.links || []).forEach((link) => {
    if (focus.has(link.target)) keep.add(link.source);
    if (focus.has(link.source)) keep.add(link.target);
  });

  let kept = nodes.filter((node) => keep.has(node.id));
  if (kept.length > limit) {
    const focused = kept.filter((node) => focus.has(node.id));
    const rest = kept.filter((node) => !focus.has(node.id));
    kept = [...focused, ...rest.slice(0, Math.max(0, limit - focused.length))];
  }

  const ids = new Set(kept.map((node) => node.id));
  const usedCategories = [...new Set(kept.map((node) => node.category))].sort((a, b) => a - b);
  const remap = new Map(usedCategories.map((category, index) => [category, index]));

  return {
    nodes: kept.map((node) => ({ ...node, category: remap.get(node.category) })),
    links: (graph.links || []).filter((link) => ids.has(link.source) && ids.has(link.target)),
    categories: usedCategories.map((category) => graph.categories[category]),
    focus: graph.focus,
  };
}

export default {
  id: 'path',
  title: '学习路径',
  subtitle: '多目标优化方案 · 帕累托前沿 · 里程碑时间轴',
  slowHint: '首次规划会运行 NSGA-II 多目标优化生成帕累托前沿，请稍候…',

  async load(ctx) {
    const [path, pareto, graph] = await Promise.all([
      api.path(ctx.learnerId, ctx.preset),
      api.pareto(ctx.learnerId, ctx.preset),
      api.graph({ learner_id: ctx.learnerId }),
    ]);
    return { path, pareto, graph };
  },

  mock(ctx) {
    return {
      path: MOCK.path(ctx.learnerId, ctx.preset),
      pareto: MOCK.pareto(ctx.learnerId, ctx.preset),
      graph: MOCK.graph({ learner_id: ctx.learnerId }),
    };
  },

  render({ path, pareto, graph }, ctx) {
    const budget = ctx.learner?.time_budget_hours_per_week ?? 8;
    const alternatives = path.alternatives || [];

    /* ---- 预算与覆盖率：直接使用算法侧权威值（docs/API.md §4），前端不做任何估算 ---- */
    const coverage = typeof path.coverage_ratio === 'number' ? path.coverage_ratio : null;
    const coveredNames = (path.covered_target_kp_names?.length
      ? path.covered_target_kp_names
      : path.covered_target_kp_ids) || [];
    const uncoveredNames = (path.uncovered_target_kp_names?.length
      ? path.uncovered_target_kp_names
      : path.uncovered_target_kp_ids) || [];
    const coverageInfo = path.coverage || {};
    const targetTotal =
      coverageInfo.target_total ?? coveredNames.length + uncoveredNames.length;
    const masteredInTarget = coverageInfo.target_mastered;

    const allPresets = [
      {
        preset: path.preset,
        preset_name: path.preset_name,
        total_hours: path.total_hours,
        weeks: path.weeks,
        total_gain: path.total_gain,
        feasibility: path.feasibility,
        coverage_ratio: coverage,
        objectives: path.objectives,
      },
      ...alternatives,
    ].sort((a, b) => ['balanced', 'fast', 'thorough'].indexOf(a.preset) - ['balanced', 'fast', 'thorough'].indexOf(b.preset));

    // 候选课程冗余度低时，不同权重可能收敛到同一组课程：给出「同解」说明，避免看起来像 bug
    const signature = (item) => `${Number(item.total_hours).toFixed(1)}|${Number(item.total_gain).toFixed(3)}`;
    const twinOf = (item) =>
      allPresets.find((other) => other.preset !== item.preset && signature(other) === signature(item));

    const presetCards = allPresets
      .map((item) => {
        const active = item.preset === path.preset;
        const twin = twinOf(item);
        return `
        <button class="preset-card ${active ? 'active' : ''}" data-preset="${esc(item.preset)}" type="button">
          <div class="preset-head">
            <span class="preset-name">${esc(item.preset_name || PRESET_LABEL[item.preset] || item.preset)}</span>
            <span class="preset-flag">● 已选</span>
            ${badge(`${item.weeks} 周`, active ? 'brand' : 'ghost')}
          </div>
          <div class="preset-nums">
            <div><span>总时长</span><strong>${esc(item.total_hours)}<small>h</small></strong></div>
            <div><span>总收益</span><strong style="color:#5ee7bc">${fx(item.total_gain, 2)}</strong></div>
            <div><span>可行性</span><strong>${pct(item.feasibility)}</strong></div>
            <div><span>覆盖率</span><strong>${item.coverage_ratio == null ? '—' : pct(item.coverage_ratio)}</strong></div>
          </div>
          <div class="preset-obj" title="四个目标的归一化得分">
            ${bar(item.objectives?.gain ?? 0, 'accent', 'thin')}
            ${bar(item.objectives?.time ?? 0, '', 'thin')}
            ${bar(item.objectives?.smoothness ?? 0, 'violet', 'thin')}
            ${bar(item.objectives?.fit ?? 0, 'warn', 'thin')}
          </div>
          ${
            twin
              ? `<div class="tiny muted" style="margin-top:8px">与「${
                  twin.preset_name || PRESET_LABEL[twin.preset]
                }」同解（候选课程冗余度低）</div>`
              : ''
          }
        </button>`;
      })
      .join('');

    const objectives = path.objectives || {};
    const overviewCard = card({
      cls: 'col-5',
      title: '方案概览',
      hint: `${path.goal?.name || ''} · ${path.preset_name || ''}`,
      body: `
        <div class="hero-ring-row" style="margin-bottom:6px">
          ${ring(path.feasibility, { size: 92, stroke: 8, color: 'var(--accent)', label: '时间可行性' })}
          <div class="hero-side-note">
            <b style="color:var(--text)">${path.weeks}</b> 周完成 ·
            共 <b style="color:var(--text)">${path.steps.length}</b> 门课<br />
            总收益 <b style="color:#5ee7bc">${fx(path.total_gain, 2)}</b> ·
            总预算 <b style="color:var(--text)">${fx(path.budget_hours ?? 0, 1)}h</b>
          </div>
        </div>
        <div class="divider"></div>
        ${[
          ['学习收益 gain', objectives.gain, 'accent'],
          ['时间成本 time', objectives.time, ''],
          ['难度平滑 smoothness', objectives.smoothness, 'violet'],
          ['风格匹配 fit', objectives.fit, 'warn'],
        ]
          .map(([label, value, tone]) =>
            `<div class="meter"><span class="meter-label">${label}</span>${bar(value ?? 0, tone)}<span class="meter-value">${fx(
              value ?? 0,
              3
            )}</span></div>`
          )
          .join('')}
        <div class="divider"></div>
        ${
          coverage == null
            ? `<p class="tiny muted">算法侧未返回覆盖率</p>`
            : `<div class="meter">
                 <span class="meter-label">目标知识点覆盖</span>
                 ${bar(coverage, coverage >= 0.8 ? 'accent' : coverage >= 0.5 ? '' : 'warn')}
                 <span class="meter-value">${pct(coverage)}</span>
               </div>`
        }
        <div class="tag-list" style="margin-top:8px">
          ${
            uncoveredNames.length
              ? `<span class="section-label" style="align-self:center">未覆盖</span>${uncoveredNames
                  .slice(0, 3)
                  .map((name) => chip(name, 'tiny'))
                  .join('')}${
                  uncoveredNames.length > 3 ? chip(`+${uncoveredNames.length - 3}`, 'tiny ghost') : ''
                }`
              : '<span class="tiny muted" style="color:#5ee7bc">全部目标知识点已被覆盖</span>'
          }
        </div>
        <div class="divider"></div>
        <div class="tag-list">
          ${(path.milestones || [])
            .slice(0, 3)
            .map((m) => chip(`W${m.week} ${m.label}`, 'tiny'))
            .join('')}
          ${
            (path.milestones || []).length > 3
              ? chip(`等 ${path.milestones.length} 个阶段`, 'tiny ghost')
              : ''
          }
        </div>`,
    });

    const paretoCard = card({
      cls: 'col-7',
      title: '帕累托前沿：时间—收益权衡',
      hint: '每个点是一条可行路径方案，金色点为当前选中方案',
      actions: chip(`${pareto.points.length} 个候选方案`, 'tiny ghost'),
      body: `<div class="chart chart-md" id="pareto-chart"></div>`,
    });

    const milestoneWeeks = [...new Set((path.milestones || []).map((m) => m.week))];
    // 引擎会为每周都给一个里程碑；为避免「处处是里程碑」，仅取首/中/末三个代表周做标记
    const milestoneMarks = new Set(
      milestoneWeeks.length > 3
        ? [milestoneWeeks[0], milestoneWeeks[Math.floor(milestoneWeeks.length / 2)], milestoneWeeks[milestoneWeeks.length - 1]]
        : milestoneWeeks
    );

    const timelineItems = path.steps
      .map((step) => {
        const milestone = (path.milestones || []).find((m) => m.week === step.week);
        const marked = milestoneMarks.has(step.week);
        return `
        <div class="tl-item ${marked ? 'milestone' : ''}">
          <div class="tl-week">第 ${step.week} 周${marked && milestone ? ` · ${esc(milestone.label)}` : ''}</div>
          <div class="tl-title">${step.order}. ${esc(step.title)}</div>
          <div class="tl-meta">
            ${chip(step.domain, 'tiny')}
            ${chip(`难度 ${step.level}`, 'tiny')}
            <span>${esc(step.hours)}h · 累计 ${esc(step.cumulative_hours)}h</span>
            ${(step.kp_names || []).slice(0, 2).map((name) => chip(name, 'tiny solid')).join('')}
          </div>
          <div class="tl-gain">
            预期增益 ${bar(step.expected_gain / 0.25, 'accent', 'thin')} <span>+${fx(step.expected_gain, 3)}</span>
          </div>
        </div>`;
      })
      .join('');

    const timelineCard = card({
      cls: 'col-7',
      title: '路径时间轴',
      hint: '按周次排布的学习步骤与里程碑',
      actions: chip(`${path.total_hours}h / ${path.weeks} 周`, 'tiny brand'),
      body: `<div class="timeline">${timelineItems || '<p class="muted tiny">暂无路径步骤</p>'}</div>`,
    });

    const weeklyCard = card({
      cls: 'col-5',
      title: '周计划负载',
      hint: '每周课程时长 vs 时间预算',
      body: `<div class="chart chart-md" id="weekly-chart"></div>`,
    });

    const local = localSubgraph(graph);
    const graphCard = card({
      cls: 'col-12',
      title: '知识图谱局部高亮',
      hint: `聚焦目标知识点与路径的一跳先修/后继（${local.nodes.length} 个节点），节点颜色 = 当前掌握度（红→绿），白边高亮 = 目标知识点，可拖拽与缩放`,
      actions: `<div class="legend-inline">
        <span><i style="background:hsl(10 70% 52%)"></i>未掌握</span>
        <span><i style="background:hsl(85 70% 52%)"></i>进行中</span>
        <span><i style="background:hsl(160 70% 52%)"></i>已掌握</span>
      </div>`,
      body: `<div class="chart chart-lg" id="path-graph"></div>`,
    });

    const kpis = `
      <div class="kpi-grid">
        ${kpi({
          label: '总时长',
          value: fx(path.total_hours, 1),
          unit: 'h',
          tone: 'accent',
          foot: `总预算 ${fx(path.budget_hours ?? 0, 1)}h · ${esc(budget)}h/周`,
        })}
        ${kpi({ label: '总收益', value: fx(path.total_gain, 2), tone: '', foot: '掌握度累计提升' })}
        ${kpi({ label: '路径课程', value: path.steps.length, unit: '门', tone: 'violet', foot: path.goal?.name || '' })}
        ${kpi({ label: '可行性', value: pct(path.feasibility), tone: path.feasibility < 0.5 ? 'danger' : 'cyan', foot: `${path.weeks} 周内完成` })}
        ${
          coverage == null
            ? ''
            : kpi({
                label: '目标覆盖率',
                value: pct(coverage),
                tone: coverage >= 0.8 ? 'accent' : 'warn',
                foot:
                  masteredInTarget == null
                    ? `${coveredNames.length}/${targetTotal || '—'} 个目标点已覆盖`
                    : `路径覆盖 ${coveredNames.length} + 已掌握 ${masteredInTarget} / ${targetTotal} 个目标点`,
              })
        }
      </div>`;

    // 预算/覆盖率异常时的防御性告警（契约保证 over_budget 恒为 false）
    const overBudgetNotice = path.over_budget
      ? `<div class="offline-banner danger">
           <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M12 7.5v5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="16.3" r="1.1"/></svg>
           <span><strong>路径超出时间预算</strong>：计划 ${fx(path.total_hours, 1)}h &gt; 预算 ${fx(
             path.budget_hours ?? 0,
             1
           )}h（算法侧标记 over_budget=true）。请缩短目标或延长周期。</span>
         </div>`
      : '';

    // 可行性过低时给出显式提示（正常输入下不应出现，作为防御保留）
    const feasibilityNotice =
      path.feasibility < 0.5
        ? `<div class="offline-banner">
             <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2.6 20h18.8L12 3Z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><path d="M12 9v5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="12" cy="17" r="1.1"/></svg>
             <span>当前方案的<strong>时间可行性仅 ${pct(path.feasibility)}</strong>：计划 ${fx(
               path.total_hours,
               1
             )} 小时 / ${path.weeks} 周，接近或超出 ${fx(path.budget_hours ?? 0, 1)} 小时的总预算。建议切换「速通方案」、缩小目标范围，或延长学习周期。</span>
           </div>`
        : '';

    return `
      ${overBudgetNotice}
      ${feasibilityNotice}
      ${kpis}
      <div class="preset-cards">${presetCards}</div>
      <div class="grid">${paretoCard}${overviewCard}</div>
      <div class="grid">${timelineCard}${weeklyCard}</div>
      <div class="grid">${graphCard}</div>`;
  },

  mount({ path, pareto, graph }, ctx) {
    const budget = ctx.learner?.time_budget_hours_per_week ?? 8;

    /* ---------------- 帕累托散点 ---------------- */
    const points = (pareto.points || []).map((point) => ({
      name: point.preset ? PRESET_LABEL[point.preset] : `候选方案 ${point.id}`,
      value: [point.hours, point.gain],
      raw: point,
      symbolSize: point.is_selected ? 18 : point.preset ? 13 : 9,
      itemStyle: {
        color: point.is_selected ? '#ffb020' : point.preset ? PALETTE[0] : 'rgba(148, 172, 224, 0.55)',
        borderColor: point.is_selected ? '#fff' : 'rgba(255,255,255,0.22)',
        borderWidth: point.is_selected ? 2.4 : 1,
        shadowBlur: point.is_selected ? 18 : 0,
        shadowColor: 'rgba(255,176,32,0.55)',
      },
      label: point.is_selected
        ? { show: true, formatter: '当前方案', position: 'top', color: '#ffb020', fontSize: 11, fontWeight: 600 }
        : { show: false },
    }));

    chartById('pareto-chart', {
      grid: { left: 12, right: 22, top: 26, bottom: 12, containLabel: true },
      tooltip: {
        trigger: 'item',
        formatter: (params) => {
          const raw = params.data.raw || {};
          return `
            <b>${raw.preset ? PRESET_LABEL[raw.preset] : '候选方案'}</b><br/>
            总时长：${raw.hours} h<br/>
            总收益：${fmt3(raw.gain)}<br/>
            课程数：${raw.n_courses} 门 · ${raw.weeks ?? '-'} 周<br/>
            平滑度：${fmt3(raw.smoothness)} · 匹配度：${fmt3(raw.fit)}`;
        },
      },
      xAxis: { type: 'value', name: '总时长 (h)', nameTextStyle: { color: '#8494b5' }, scale: true },
      yAxis: { type: 'value', name: '总收益', nameTextStyle: { color: '#8494b5' }, scale: true },
      series: [
        {
          type: 'scatter',
          data: points,
          emphasis: { scale: 1.35, itemStyle: { borderWidth: 2 } },
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: 'rgba(148,172,224,0.28)', type: 'dashed' },
            label: { color: '#8494b5', fontSize: 10 },
            data: [{ xAxis: budget * (path.weeks || 1), name: '预算上限' }],
          },
        },
      ],
    });

    /* ---------------- 周计划 ---------------- */
    const weeks = (path.weekly_plan || []).map((row) => row.week);
    const manyWeeks = weeks.length > 24;
    chartById('weekly-chart', {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { top: 0, data: ['计划时长', '时间预算'] },
      grid: { left: 10, right: 16, top: 40, bottom: manyWeeks ? 34 : 6, containLabel: true },
      xAxis: {
        type: 'category',
        data: weeks.map((w) => `W${w}`),
        axisLabel: { rotate: manyWeeks ? 45 : 0, interval: manyWeeks ? 'auto' : 0, fontSize: 10 },
      },
      yAxis: { type: 'value', name: '小时', nameTextStyle: { color: '#8494b5' } },
      dataZoom: manyWeeks
        ? [
            { type: 'inside' },
            { type: 'slider', height: 12, bottom: 8, borderColor: 'transparent' },
          ]
        : [],
      series: [
        {
          name: '计划时长',
          type: 'bar',
          data: (path.weekly_plan || []).map((row) => row.hours),
          itemStyle: { color: PALETTE[0], borderRadius: [4, 4, 0, 0] },
          barMaxWidth: 22,
        },
        {
          name: '时间预算',
          type: 'line',
          data: weeks.map(() => budget),
          smooth: true,
          symbol: 'none',
          lineStyle: { color: PALETTE[1], width: 1.6, type: 'dashed' },
          itemStyle: { color: PALETTE[1] },
        },
      ],
    });

    /* ---------------- 知识图谱（局部聚焦） ---------------- */
    const local = localSubgraph(graph);
    const categories = (local.categories || []).map((name) => ({ name }));
    const targetSet = new Set(local.focus?.target_kps || []);
    const nodes = (local.nodes || []).map((node) => {
      const highlighted = targetSet.has(node.id);
      return {
        id: node.id,
        name: node.name,
        category: node.category,
        symbolSize: highlighted ? node.symbolSize + 7 : node.symbolSize,
        value: node.mastery,
        itemStyle: {
          color: masteryColor(node.mastery),
          borderColor: highlighted ? '#ffffff' : PALETTE[node.category % PALETTE.length],
          borderWidth: highlighted ? 2.6 : 1.2,
          shadowBlur: highlighted ? 14 : 0,
          shadowColor: 'rgba(47,107,255,0.5)',
        },
        label: {
          show: highlighted || node.difficulty <= 1,
          fontSize: 10,
          color: '#cfdcf7',
          position: 'right',
          formatter: () => node.name,
        },
      };
    });

    chartById('path-graph', {
      tooltip: {
        formatter: (params) => {
          if (params.dataType === 'edge') return `${params.data.source} → ${params.data.target}`;
          const mastery = params.data.value;
          return `<b>${params.data.name}</b><br/>${params.data.id}<br/>掌握度：${
            mastery == null ? '未记录' : fmt3(mastery)
          }`;
        },
      },
      legend: [{ data: categories.map((c) => c.name), top: 0, type: 'scroll' }],
      series: [
        {
          type: 'graph',
          layout: 'force',
          roam: true,
          draggable: true,
          categories,
          data: nodes,
          links: (local.links || []).map((link) => ({
            source: link.source,
            target: link.target,
            lineStyle: { width: 1, opacity: 0.55, curveness: 0.12 },
          })),
          force: { repulsion: 220, edgeLength: [38, 96], gravity: 0.08 },
          emphasis: { focus: 'adjacency', lineStyle: { width: 2, opacity: 0.9 } },
          labelLayout: { hideOverlap: true },
          scaleLimit: { min: 0.5, max: 3.5 },
        },
      ],
    });
  },

  bind(root, ctx) {
    root.querySelectorAll('.preset-card').forEach((node) => {
      node.addEventListener('click', () => {
        const preset = node.getAttribute('data-preset');
        if (preset && preset !== ctx.preset) ctx.setPreset(preset);
      });
    });
  },
};
