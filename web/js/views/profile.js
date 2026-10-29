/**
 * 视图 1：学习者画像
 * 基本信息、能力雷达、领域掌握度、目标知识点缺口、时间预算、学习历史。
 */

import { api } from '../core/api.js';
import { chartById } from '../core/charts.js';
import { MOCK } from '../mock.js';
import {
  bar,
  card,
  chip,
  esc,
  fx,
  hrs,
  initials,
  kpi,
  meterRow,
  pct,
  ring,
} from '../core/ui.js';

const shortDomain = (name) => (name.length > 6 ? `${name.slice(0, 5)}…` : name);

/** 缺口表最多展示条数（完整数据集下缺口可达数十个，全量铺开会把页面撑得很长） */
const GAP_TABLE_LIMIT = 12;

const gapTone = (value) => (value < 0.3 ? 'danger' : value < 0.6 ? 'warn' : 'accent');

export default {
  id: 'profile',
  title: '学习者画像',
  subtitle: '能力掌握度、目标缺口与时间预算',

  async load(ctx) {
    const [detail, graph] = await Promise.all([
      api.learner(ctx.learnerId),
      api.graph({ learner_id: ctx.learnerId }),
    ]);
    return { detail, graph };
  },

  mock(ctx) {
    return { detail: MOCK.learnerDetail(ctx.learnerId), graph: MOCK.graph({ learner_id: ctx.learnerId }) };
  },

  render({ detail }, ctx) {
    const { learner, goal, profile, history } = detail;
    const style = learner.learning_style || {};
    const masteryNames = profile.mastery_names || [];
    const topMastery = masteryNames.slice(0, 10);
    const domains = Object.keys(profile.domain_mastery || {});
    const gapEntries = Object.entries(profile.gaps || {}).sort((a, b) => Number(b[1]) - Number(a[1]));

    const heroCards = `
      ${card({
        body: `
          <div class="hero-identity">
            <div class="avatar lg">${initials(learner.nickname)}</div>
            <div style="min-width:0;flex:1">
              <div class="hero-name">
                <h2>${esc(learner.nickname)}</h2>
                ${chip(learner.learner_id, 'tiny')}
                ${chip(goal.name, 'tiny solid')}
              </div>
              <div class="hero-meta">
                <span>动机 <b>${esc(learner.motivation)}</b></span>
                <span>${esc(learner.background?.education || '')} · ${esc(learner.background?.major || '')}</span>
                <span>当前角色 <b>${esc(learner.background?.current_role || '')}</b></span>
                <span>周期 <b>${esc(learner.deadline_weeks)}</b> 周</span>
              </div>
              <div class="hero-ability">
                ${meterRow('综合能力', learner.ability, { display: fx(learner.ability, 2) })}
                ${meterRow('目标就绪度', profile.readiness, { display: fx(profile.readiness, 2), tone: 'accent' })}
                ${meterRow('历史完成率', profile.completion_rate, {
                  display: pct(profile.completion_rate),
                  tone: 'violet',
                })}
              </div>
              <div class="divider" style="margin:10px 0 8px"></div>
              <div class="tag-list">
                <span class="section-label" style="align-self:center">目标知识点</span>
                ${(profile.target_kp_names || [])
                  .slice(0, 6)
                  .map((name) => chip(name, 'tiny'))
                  .join('')}
                ${
                  (profile.target_kp_names || []).length > 6
                    ? chip(`+${profile.target_kp_names.length - 6}`, 'tiny ghost')
                    : ''
                }
              </div>
            </div>
          </div>`,
      })}

      ${card({
        title: '时间预算',
        hint: '每周可投入时长 × 剩余周期',
        body: `
          <div class="budget-grid">
            <div class="budget-item"><span>每周投入</span><strong>${esc(
              learner.time_budget_hours_per_week
            )}<small>h</small></strong></div>
            <div class="budget-item"><span>剩余周期</span><strong>${esc(
              learner.deadline_weeks
            )}<small>周</small></strong></div>
            <div class="budget-item"><span>总可投入</span><strong>${esc(
              profile.total_available_hours
            )}<small>h</small></strong></div>
            <div class="budget-item"><span>单次专注</span><strong>${esc(
              style.session_len_min ?? 45
            )}<small>min</small></strong></div>
          </div>
          <div class="divider"></div>
          <div class="tag-list">
            ${chip(`偏好模态 ${style.prefer_modality === 'project' ? '项目实战' : style.prefer_modality === 'text' ? '图文课' : style.prefer_modality === 'mixed' ? '混合式' : '视频课'}`, 'tiny solid')}
            ${chip(`每周 ${style.weekly_days ?? 4} 天`, 'tiny')}
            ${chip(`理论偏好 ${fx(style.theory_preference ?? 0.5, 2)}`, 'tiny')}
          </div>`,
      })}

      ${card({
        title: '学习进度',
        body: `
          <div class="hero-ring-row">
            ${ring(profile.readiness, {
              size: 96,
              stroke: 8,
              color: 'var(--accent)',
              label: '就绪度',
            })}
            <div class="hero-side-note">
              目标知识点 <b style="color:var(--text)">${profile.target_kps.length}</b> 个<br />
              已掌握 <b style="color:#5ee7bc">${profile.mastered_count}</b> ·
              缺口 <b style="color:#ffcb6b">${profile.gap_count}</b>
            </div>
          </div>
          <div class="divider"></div>
          ${meterRow('目标覆盖率', profile.readiness, { display: fx(profile.readiness, 2), tone: 'accent' })}`,
      })}
    `;

    const radarCard = card({
      cls: 'col-4',
      title: '能力雷达',
      hint: '七领域掌握度 vs 目标水平',
      body: `<div class="chart chart-md" id="profile-radar"></div>`,
    });

    const domainCard = card({
      cls: 'col-3',
      title: '领域掌握度',
      hint: '按知识点掌握度平均值',
      body:
        domains.map((domain) => meterRow(domain, profile.domain_mastery[domain], { display: fx(profile.domain_mastery[domain], 2) })).join('') ||
        `<p class="muted tiny">暂无数据</p>`,
    });

    const gapRows = gapEntries
      .slice(0, GAP_TABLE_LIMIT)
      .map(([kpId, gapValue]) => {
        const name = profile.gap_names?.[gapEntries.findIndex(([id]) => id === kpId)] ?? kpId;
        const mastery = profile.mastery[kpId] ?? 0;
        return `
        <tr>
          <td class="cell-strong">${esc(name)}</td>
          <td class="cell-muted">${esc(kpId)}</td>
          <td><div class="cell-bar">${bar(mastery, gapTone(mastery))}<span>${fx(mastery, 2)}</span></div></td>
          <td class="num" style="color:${gapValue > 0.4 ? 'var(--danger)' : 'var(--warn)'}">-${fx(gapValue, 2)}</td>
          <td>${chip(gapValue > 0.4 ? '优先补强' : '可小幅补齐', gapValue > 0.4 ? 'tiny' : 'tiny solid')}</td>
        </tr>`;
      })
      .join('');

    const gapCard = card({
      cls: 'col-7',
      title: '目标知识点缺口',
      hint: '按缺口幅度降序，展示前 ' + GAP_TABLE_LIMIT + ' 条',
      actions: `${chip(`共 ${gapEntries.length} 个缺口`, 'tiny warn')} ${chip(
        `${profile.target_kps.length} 个目标点`,
        'tiny ghost'
      )}`,
      bodyCls: 'flush',
      body: gapEntries.length
        ? `<div class="table-wrap">
             <table class="table">
               <thead><tr><th>知识点</th><th>编号</th><th>当前掌握度</th><th class="num">缺口</th><th>建议</th></tr></thead>
               <tbody>${gapRows}</tbody>
             </table>
           </div>`
        : `<div style="padding:18px"><p class="muted tiny">该学习者暂无明显知识缺口。</p></div>`,
    });

    const kpis = `
      <div class="kpi-grid">
        ${kpi({ label: '已掌握知识点', value: profile.mastered_count, unit: '个', tone: 'accent', foot: '目标点中掌握度 ≥ 0.85' })}
        ${kpi({ label: '知识缺口', value: profile.gap_count, unit: '个', tone: 'warn', foot: '目标点中掌握度 < 0.60' })}
        ${kpi({ label: '目标就绪度', value: fx(profile.readiness, 2), tone: '', foot: `${goal.name} 岗位要求` })}
        ${kpi({ label: '历史完成率', value: pct(profile.completion_rate), tone: 'violet', foot: `共 ${history.length} 条学习记录` })}
        ${kpi({ label: '总可投入时长', value: esc(profile.total_available_hours), unit: 'h', tone: 'cyan', foot: `${esc(learner.time_budget_hours_per_week)} h/周` })}
      </div>`;

    const topBars = topMastery
      .map((item) =>
        meterRow(item.name, item.value, {
          display: fx(item.value, 2),
          tone: item.value >= 0.85 ? 'accent' : item.value < 0.6 ? 'warn' : '',
          title: `${item.kp_id} · ${item.name}`,
        })
      )
      .join('');

    const historyRows = history
      .map(
        (item) => `
        <div class="history-item ${item.completed ? 'done' : ''}">
          <i class="marker"></i>
          <div style="min-width:0">
            <div class="history-title">${esc(item.title)}</div>
            <div class="history-sub">
              <span>${esc(item.domain)}</span>
              <span>进度 ${pct(item.progress)}</span>
              <span>${hrs(item.hours_spent)}</span>
              ${item.quiz_score != null ? `<span>测验 ${fx(item.quiz_score, 2)}</span>` : ''}
              ${item.rating != null ? `<span>评分 ${esc(item.rating)}★</span>` : ''}
            </div>
          </div>
          <div class="history-right">
            ${item.completed ? chip('已完成', 'tiny solid') : chip('进行中', 'tiny ghost')}
            <div style="margin-top:4px">${esc(String(item.timestamp).slice(0, 10))}</div>
          </div>
        </div>`
      )
      .join('');

    return `
      <div class="grid hero">${heroCards}</div>
      ${kpis}
      <div class="grid">
        ${radarCard}
        ${domainCard}
        ${card({
          cls: 'col-5',
          title: '掌握度最高的知识点',
          hint: 'Top 10（按掌握度降序）',
          body: topBars || `<p class="muted tiny">暂无掌握度记录</p>`,
        })}
      </div>
      <div class="grid">
        ${gapCard}
        ${card({
          cls: 'col-5',
          title: '学习历史',
          hint: '最近学习记录（新 → 旧）',
          actions: chip(`${history.length} 条`, 'tiny ghost'),
          body: `<div class="history-list">${historyRows || '<p class="muted tiny">暂无学习记录</p>'}</div>`,
        })}
      </div>
    `;
  },

  mount({ detail }) {
    const domains = Object.keys(detail.profile.domain_mastery || {});
    const target = 0.8;
    chartById('profile-radar', {
      tooltip: { trigger: 'item' },
      legend: { bottom: 0, data: ['当前掌握', '目标水平'] },
      radar: {
        indicator: domains.map((name) => ({ name: shortDomain(name), max: 1 })),
        radius: '64%',
        center: ['50%', '48%'],
      },
      series: [
        {
          type: 'radar',
          symbolSize: 5,
          data: [
            {
              name: '当前掌握',
              value: domains.map((d) => detail.profile.domain_mastery[d]),
              areaStyle: { opacity: 0.26 },
              lineStyle: { width: 2 },
            },
            {
              name: '目标水平',
              value: domains.map(() => target),
              lineStyle: { type: 'dashed', width: 1.4 },
              areaStyle: { opacity: 0.04 },
            },
          ],
        },
      ],
    });
  },
};
