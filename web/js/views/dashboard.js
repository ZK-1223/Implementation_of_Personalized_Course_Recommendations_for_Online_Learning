/**
 * 视图 4：学习分析仪表板
 * KPI、掌握度曲线、领域雷达、难度分布、模态构成、时间分配、完成概率预测、缺口明细。
 */

import { api } from '../core/api.js';
import { chartById, PALETTE, fmt3, gradient } from '../core/charts.js';
import { MOCK } from '../mock.js';
import { bar, badge, card, chip, esc, fx, kpi, meterRow, pct } from '../core/ui.js';

const riskTone = (level) => (level === '低' ? 'accent' : level === '中' ? 'warn' : 'danger');
const gapTone = (value) => (value < 0.3 ? 'danger' : value < 0.6 ? 'warn' : 'accent');

export default {
  id: 'dashboard',
  title: '学习分析',
  subtitle: '进度 KPI · 掌握度演化 · 时间分配 · 完成概率预测',

  async load(ctx) {
    return api.analytics(ctx.learnerId, ctx.preset);
  },

  mock(ctx) {
    return MOCK.analytics(ctx.learnerId);
  },

  render(data) {
    const kpiData = data.kpi || {};
    const forecast = data.completion_forecast || [];
    const gaps = data.gap_table || [];
    const risky = forecast.filter((row) => row.risk_level !== '低');
    const inPathGaps = gaps.filter((row) => row.in_path).length;

    const kpis = `
      <div class="kpi-grid">
        ${kpi({
          label: '总体进度',
          value: pct(kpiData.overall_progress),
          tone: '',
          foot: `掌握 ${kpiData.mastered_kps}/${kpiData.total_target_kps} 个目标点`,
        })}
        ${kpi({ label: '平均掌握度', value: fx(kpiData.avg_mastery, 3), tone: 'accent', foot: '全部已记录知识点' })}
        ${kpi({ label: '目标就绪度', value: fx(kpiData.readiness, 2), tone: 'cyan', foot: '可直接开始目标课程' })}
        ${kpi({
          label: '规划时长',
          value: fx(kpiData.planned_hours, 1),
          unit: 'h',
          tone: 'violet',
          foot: `可用 ${fx(kpiData.available_hours, 1)}h`,
        })}
        ${kpi({ label: '风险课程', value: kpiData.risk_courses ?? risky.length, unit: '门', tone: 'danger', foot: '完成概率 < 0.70' })}
        ${kpi({ label: '连续学习', value: kpiData.learning_streak_weeks ?? 0, unit: '周', tone: 'warn', foot: '保持节奏很重要' })}
      </div>`;

    const masteryCard = card({
      cls: 'col-7',
      title: '掌握度演化曲线',
      hint: '按学习周次回放的平均掌握度与已掌握知识点数',
      body: `<div class="chart chart-md" id="dash-mastery"></div>`,
    });

    const radarCard = card({
      cls: 'col-5',
      title: '领域掌握雷达',
      hint: '各领域当前掌握度 vs 目标要求（0.80）',
      body: `<div class="chart chart-md" id="dash-radar"></div>`,
    });

    const timeCard = card({
      cls: 'col-7',
      title: '每周时间分配',
      hint: '实际投入 vs 计划预算（柱 + 线组合，悬停可看当周完成课程数）',
      body: `<div class="chart chart-md" id="dash-time"></div>`,
    });

    const modalityCard = card({
      cls: 'col-5',
      title: '模态构成',
      hint: '路径课程按承载形式的学习时长占比',
      body: `<div class="chart chart-md" id="dash-modality"></div>`,
    });

    const difficultyCard = card({
      cls: 'col-4',
      title: '难度分布',
      hint: '路径内课程按难度等级计数（门）',
      body: `<div class="chart chart-sm" id="dash-difficulty"></div>`,
    });

    const forecastCard = card({
      cls: 'col-8',
      title: '完成概率预测',
      hint: '效果预测模型对路径内课程的结课概率预测',
      actions: chip(`${forecast.length} 门课`, 'tiny ghost'),
      bodyCls: 'flush',
      body: `
        <div class="table-wrap">
          <table class="table">
            <thead>
              <tr>
                <th class="num">#</th><th>课程</th><th>完成概率</th><th>风险</th>
              </tr>
            </thead>
            <tbody>
              ${forecast
                .map(
                  (row) => `
                <tr>
                  <td class="num cell-muted">${row.order}</td>
                  <td class="cell-strong">${esc(row.course_title || row.course_id)}</td>
                  <td><div class="cell-bar">${bar(row.p_completion, riskTone(row.risk_level))}<span>${pct(
                    row.p_completion
                  )}</span></div></td>
                  <td>${badge(row.risk_level, riskTone(row.risk_level))}</td>
                </tr>`
                )
                .join('')}
            </tbody>
          </table>
        </div>`,
    });

    const gapCard = card({
      cls: 'col-8',
      title: '知识缺口明细',
      hint: '目标知识点中掌握度不足的部分',
      actions: chip(`${inPathGaps} 个已排入路径`, 'tiny accent'),
      bodyCls: 'flush',
      body: `
        <div class="table-wrap">
          <table class="table">
            <thead>
              <tr><th>知识点</th><th>领域</th><th class="num">难度</th><th>掌握度</th><th>路径覆盖</th></tr>
            </thead>
            <tbody>
              ${gaps
                .map(
                  (row) => `
                <tr>
                  <td class="cell-strong">${esc(row.name)}<div class="tiny muted">${esc(row.kp_id)}</div></td>
                  <td class="cell-muted">${esc(row.domain)}</td>
                  <td class="num">${esc(row.difficulty)}</td>
                  <td><div class="cell-bar">${bar(row.mastery, gapTone(row.mastery))}<span>${fx(
                    row.mastery,
                    2
                  )}</span></div></td>
                  <td>${row.in_path ? badge('已安排', 'accent') : badge('待安排', 'ghost')}</td>
                </tr>`
                )
                .join('')}
            </tbody>
          </table>
        </div>`,
    });

    const insightCard = card({
      cls: 'col-4',
      title: '关键洞察',
      hint: '由当前数据自动汇总',
      body: `
        ${meterRow('路径时间占预算', Math.min(1, (kpiData.planned_hours || 0) / Math.max(1, kpiData.available_hours || 1)), {
          display: pct(Math.min(1, (kpiData.planned_hours || 0) / Math.max(1, kpiData.available_hours || 1))),
        })}
        <div class="divider"></div>
        <ul class="notes-list">
          <li>当前平均掌握度 ${fx(kpiData.avg_mastery, 3)}，目标就绪度 ${fx(kpiData.readiness, 2)}。</li>
          ${
            risky.length
              ? `<li>有 <b style="color:#ffcb6b">${risky.length}</b> 门课程完成概率偏低，建议拆分学习时段或延长周期。</li>`
              : '<li>路径内课程完成概率均在安全区间，保持当前节奏即可。</li>'
          }
          <li>${inPathGaps} 个知识缺口已排入路径，建议优先完成对应课程。</li>
          <li>可用时长 ${fx(kpiData.available_hours, 1)}h，规划占用 ${fx(kpiData.planned_hours, 1)}h。</li>
        </ul>`,
    });

    return `
      ${kpis}
      <div class="grid">${masteryCard}${radarCard}</div>
      <div class="grid">${timeCard}${modalityCard}</div>
      <div class="grid">${difficultyCard}${forecastCard}</div>
      <div class="grid">${gapCard}${insightCard}</div>`;
  },

  mount(data) {
    /* ---------------- 掌握度曲线 ---------------- */
    const curve = data.mastery_curve || [];
    chartById('dash-mastery', {
      tooltip: { trigger: 'axis' },
      legend: { top: 0, data: ['平均掌握度', '已掌握知识点'] },
      grid: { left: 10, right: 22, top: 42, bottom: 6, containLabel: true },
      xAxis: { type: 'category', data: curve.map((row) => `W${row.week}`) },
      yAxis: [
        { type: 'value', max: 1, name: '掌握度', axisLabel: { formatter: (v) => `${Math.round(v * 100)}%` } },
        { type: 'value', name: '个数', splitLine: { show: false } },
      ],
      series: [
        {
          name: '平均掌握度',
          type: 'line',
          smooth: true,
          symbolSize: 7,
          data: curve.map((row) => row.avg_mastery),
          lineStyle: { width: 2.6, color: PALETTE[0] },
          itemStyle: { color: PALETTE[0] },
          areaStyle: { color: gradient(PALETTE[0], 0.36, 0.02) },
        },
        {
          name: '已掌握知识点',
          type: 'line',
          yAxisIndex: 1,
          smooth: true,
          symbol: 'circle',
          symbolSize: 5,
          data: curve.map((row) => row.mastered),
          lineStyle: { width: 1.8, type: 'dashed', color: PALETTE[1] },
          itemStyle: { color: PALETTE[1] },
        },
      ],
    });

    /* ---------------- 领域雷达 ---------------- */
    const radar = data.domain_radar || [];
    chartById('dash-radar', {
      tooltip: { trigger: 'item' },
      legend: { bottom: 0, data: ['当前掌握', '目标要求'] },
      radar: {
        indicator: radar.map((row) => ({
          name: row.domain.length > 6 ? `${row.domain.slice(0, 5)}…` : row.domain,
          max: 1,
        })),
        radius: '62%',
        center: ['50%', '46%'],
      },
      series: [
        {
          type: 'radar',
          data: [
            {
              name: '当前掌握',
              value: radar.map((row) => row.mastery),
              areaStyle: { opacity: 0.3 },
              lineStyle: { width: 2 },
            },
            {
              name: '目标要求',
              value: radar.map((row) => row.target),
              lineStyle: { type: 'dashed', width: 1.3 },
              areaStyle: { opacity: 0.05 },
            },
          ],
        },
      ],
    });

    /* ---------------- 时间分配 ---------------- */
    const allocation = data.time_allocation || [];
    const activity = data.history_activity || [];
    const actualMap = Object.fromEntries(activity.map((row) => [row.week, row.hours]));
    const completedMap = Object.fromEntries(activity.map((row) => [row.week, row.completed]));
    const manyWeeks = allocation.length > 24;
    chartById('dash-time', {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params) => {
          const index = params?.[0]?.dataIndex ?? 0;
          const row = allocation[index] || {};
          return [
            `<b>第 ${row.week} 周</b>`,
            `实际投入：${actualMap[row.week] ?? 0} h`,
            `计划预算：${row.planned ?? 0} h`,
            `完成课程：${completedMap[row.week] ?? 0} 门`,
          ].join('<br/>');
        },
      },
      legend: { top: 0, data: ['实际投入', '计划预算'] },
      grid: { left: 10, right: 22, top: 42, bottom: manyWeeks ? 34 : 6, containLabel: true },
      xAxis: {
        type: 'category',
        data: allocation.map((row) => `W${row.week}`),
        axisLabel: { rotate: manyWeeks ? 45 : 0, interval: manyWeeks ? 'auto' : 0, fontSize: 10 },
      },
      dataZoom: manyWeeks
        ? [{ type: 'inside' }, { type: 'slider', height: 12, bottom: 8, borderColor: 'transparent' }]
        : [],
      yAxis: [{ type: 'value', name: '小时', splitLine: { lineStyle: { type: 'dashed' } } }],
      series: [
        {
          name: '实际投入',
          type: 'bar',
          data: allocation.map((row) => actualMap[row.week] ?? 0),
          itemStyle: { color: PALETTE[0], borderRadius: [4, 4, 0, 0] },
          barMaxWidth: 20,
        },
        {
          name: '计划预算',
          type: 'line',
          data: allocation.map((row) => row.planned),
          smooth: true,
          symbol: 'none',
          lineStyle: { color: PALETTE[1], width: 1.8, type: 'dashed' },
        },
      ],
    });

    /* ---------------- 模态构成 ---------------- */
    const mix = (data.modality_mix || []).filter((row) => Number(row.hours ?? 0) > 0);
    chartById('dash-modality', {
      tooltip: { trigger: 'item', formatter: '{b}<br/>{c} h（{d}%）' },
      legend: { bottom: 0, itemGap: 12 },
      series: [
        {
          type: 'pie',
          radius: ['52%', '74%'],
          center: ['50%', '45%'],
          avoidLabelOverlap: true,
          itemStyle: { borderColor: 'rgba(9,14,26,0.9)', borderWidth: 2, borderRadius: 5 },
          label: { show: true, formatter: '{b}\n{d}%', color: '#b9c6e2', fontSize: 10.5, lineHeight: 15 },
          labelLine: { length: 8, length2: 8, lineStyle: { color: 'rgba(148,172,224,0.3)' } },
          data: mix.map((row) => ({ name: row.name || row.modality, value: row.hours })),
        },
      ],
    });

    /* ---------------- 难度分布 ---------------- */
    const difficulty = data.difficulty_distribution || [];
    chartById('dash-difficulty', {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 6, right: 12, top: 20, bottom: 4, containLabel: true },
      xAxis: {
        type: 'category',
        data: difficulty.map((row) => `L${row.level}`),
      },
      yAxis: { type: 'value', minInterval: 1, splitLine: { lineStyle: { type: 'dashed' } } },
      series: [
        {
          type: 'bar',
          data: difficulty.map((row, index) => ({
            value: row.count,
            itemStyle: {
              color: PALETTE[index % PALETTE.length],
              borderRadius: [5, 5, 0, 0],
            },
          })),
          barMaxWidth: 26,
          label: { show: true, position: 'top', color: '#8494b5', fontSize: 10 },
        },
      ],
    });
  },
};
