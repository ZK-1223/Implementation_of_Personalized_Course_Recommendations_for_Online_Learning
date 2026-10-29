/**
 * 视图 5：社交学习
 * 相似同伴（相似度 / 共同目标 / 共同课程 / 对方更擅长的知识点）与互补型学习小组。
 */

import { api } from '../core/api.js';
import { MOCK } from '../mock.js';
import { bar, badge, card, chip, esc, fx, initials, kpi, pct } from '../core/ui.js';

export default {
  id: 'social',
  title: '社交学习',
  subtitle: '相似同伴匹配 · 互补学习小组 · 同期群对照',

  async load(ctx) {
    return api.social(ctx.learnerId, ctx.params.k ?? 6);
  },

  mock(ctx) {
    return MOCK.social(ctx.learnerId, ctx.params.k ?? 6);
  },

  render(data, ctx) {
    const cohort = data.cohort_summary || {};
    const peers = data.peers || [];
    const groups = data.groups || [];
    const yourName = ctx.learner?.nickname || '我';

    const kpis = `
      <div class="kpi-grid">
        ${kpi({ label: '相似同伴', value: peers.length, unit: '位', tone: '', foot: '按画像+行为相似度排序' })}
        ${kpi({ label: '平均相似度', value: fx(cohort.avg_similarity, 3), tone: 'accent', foot: '余弦相似度均值' })}
        ${kpi({ label: '同期群完成率', value: pct(cohort.cohort_completion_rate), tone: 'cyan', foot: '同伴群体均值' })}
        ${kpi({ label: '我的完成率', value: pct(cohort.your_completion_rate), tone: 'violet', foot: '与同期群对照' })}
        ${kpi({ label: '可加入小组', value: groups.length, unit: '个', tone: 'warn', foot: '按知识缺口聚类' })}
      </div>`;

    const peerCards = peers
      .map(
        (peer, index) => `
        <article class="peer-card" style="animation:view-in .45s var(--ease) both;animation-delay:${index * 55}ms">
          <div class="peer-head">
            <div class="avatar">${initials(peer.nickname)}</div>
            <div style="min-width:0">
              <div class="peer-name">${esc(peer.nickname)}</div>
              <div class="peer-sub">${esc(peer.goal_name)} · 能力 ${fx(peer.ability, 2)} · 已完成 ${
          peer.completed_count ?? 0
        } 门</div>
            </div>
            <div class="peer-sim">
              <strong>${fx(peer.similarity, 2)}</strong>
              <span>相似度</span>
            </div>
          </div>

          <div class="peer-section">
            <span class="section-label">共同目标知识点</span>
            <div class="tag-list">
              ${(peer.shared_goal_kps || []).slice(0, 4).map((name) => chip(name, 'tiny solid')).join('') ||
                '<span class="tiny muted">暂无</span>'}
            </div>
          </div>

          <div class="peer-section">
            <span class="section-label">共同课程</span>
            <div class="tag-list">
              ${(peer.shared_courses || []).slice(0, 3).map((name) => chip(name, 'tiny')).join('') ||
                '<span class="tiny muted">暂无</span>'}
            </div>
          </div>

          <div class="peer-section">
            <span class="section-label">对方更擅长</span>
            <div class="tag-list">
              ${(peer.ahead_in || []).slice(0, 3).map((name) => badge(name, 'accent')).join('') ||
                '<span class="tiny muted">—</span>'}
            </div>
          </div>
        </article>`
      )
      .join('');

    const groupCards = groups
      .map(
        (group) => `
        <article class="group-card">
          <div class="peer-head">
            <span class="badge badge-brand">${esc(group.group_id)}</span>
            <div class="tag-list" style="flex:1">
              ${(group.kp_names || []).slice(0, 3).map((name) => chip(name, 'tiny solid')).join('')}
            </div>
          </div>
          <div class="group-members">
            ${(group.members || [])
              .map(
                (member) => `
                <span class="member ${member.nickname === yourName ? 'is-you' : ''}">
                  <span class="avatar sm">${initials(member.nickname)}</span>
                  ${esc(member.nickname)} · ${fx(member.ability, 2)}
                </span>`
              )
              .join('')}
          </div>
          <div class="meter">
            <span class="meter-label">平均完成率</span>
            ${bar(group.avg_completion, '', 'thin')}
            <span class="meter-value">${pct(group.avg_completion)}</span>
          </div>
          <div class="meter">
            <span class="meter-label">协同度 synergy</span>
            ${bar(group.synergy, 'accent', 'thin')}
            <span class="meter-value">${fx(group.synergy, 2)}</span>
          </div>
        </article>`
      )
      .join('');

    return `
      ${kpis}
      <div class="grid">
        ${card({
          cls: 'col-12',
          title: '同伴匹配结果',
          hint: '相似度基于掌握度向量余弦相似度 + 共同目标/课程的行为重合度',
          actions: chip(`共 ${peers.length} 位`, 'tiny ghost'),
          body: `<div class="peer-grid">${peerCards || '<p class="muted tiny">暂无可匹配同伴</p>'}</div>`,
        })}
      </div>
      <div class="grid">
        ${card({
          cls: 'col-12',
          title: '互补型学习小组',
          hint: '以知识缺口为共识主题，按能力互补性组队（协同度越高越互补）',
          body: `<div class="group-grid">${groupCards || '<p class="muted tiny">暂无可组建小组</p>'}</div>`,
        })}
      </div>`;
  },
};
