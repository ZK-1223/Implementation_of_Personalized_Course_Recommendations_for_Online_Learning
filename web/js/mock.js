/**
 * 离线示例数据（契约示例）。
 *
 * 作用：后端算法模块尚未就绪、或演示环境断网时，前端仍可完整呈现 6 个视图。
 * 所有结构严格对齐 docs/API.md，字段名与类型与真实接口一致，
 * 因此视图层代码只有一条渲染路径，不需要区分「真实 / 示例」数据。
 *
 * 数据规模参照 learnpath/data/fixtures.py 的微型数据集：7 领域 × 5 知识点 = 35 个知识点。
 */

const DOMAINS = ['编程开发', '数据与人工智能', '前端与设计', '网络与安全', '产品与职场', '语言学习', '数学基础'];

const DOMAIN_ICONS = {
  编程开发: 'code',
  数据与人工智能: 'chart',
  前端与设计: 'layout',
  网络与安全: 'shield',
  产品与职场: 'briefcase',
  语言学习: 'language',
  数学基础: 'function',
};

const KP_NAMES = {
  编程开发: ['变量与数据类型', '流程控制', '函数与模块', '面向对象', '并发编程'],
  数据与人工智能: ['数据清洗', '描述性统计', '假设检验', '回归建模', '模型评估'],
  前端与设计: ['HTML 语义化', 'CSS 布局', 'JavaScript 基础', '组件化开发', '性能优化'],
  网络与安全: ['网络分层模型', 'TCP 与 UDP', 'Web 攻防基础', '加密与证书', '渗透测试'],
  产品与职场: ['需求调研', '用户故事', '原型设计', '数据分析看板', '增长实验'],
  语言学习: ['音标与连读', '核心词汇', '基础句型', '学术写作', '商务表达'],
  数学基础: ['函数与极限', '线性代数', '概率论', '最优化基础', '数值计算'],
};

const MODALITY_NAMES = { video: '视频课', text: '图文课', project: '项目实战', mixed: '混合式' };

/** 35 个知识点（id 规则与 fixtures 一致：KP-{领域序号}-{难度序号}） */
const KPS = DOMAINS.flatMap((domain, dIndex) =>
  KP_NAMES[domain].map((name, kIndex) => ({
    id: `KP-${String(dIndex + 1).padStart(2, '0')}-${String(kIndex + 1).padStart(2, '0')}`,
    name,
    domain,
    difficulty: kIndex + 1,
    depth: kIndex,
  }))
);

const KP_BY_ID = Object.fromEntries(KPS.map((kp) => [kp.id, kp]));
const KP_BY_NAME = Object.fromEntries(KPS.map((kp) => [kp.name, kp]));

/** 领域内先修链 + 3 条跨领域先修边 */
const EDGES = (() => {
  const list = [];
  DOMAINS.forEach((domain, dIndex) => {
    for (let i = 0; i < 4; i += 1) {
      list.push({
        source: `KP-${String(dIndex + 1).padStart(2, '0')}-${String(i + 1).padStart(2, '0')}`,
        target: `KP-${String(dIndex + 1).padStart(2, '0')}-${String(i + 2).padStart(2, '0')}`,
      });
    }
  });
  const cross = [
    ['函数与模块', '数据清洗'],
    ['概率论', '假设检验'],
    ['描述性统计', '数据分析看板'],
  ];
  cross.forEach(([a, b]) => list.push({ source: KP_BY_NAME[a].id, target: KP_BY_NAME[b].id }));
  return list;
})();

/** 课程库（14 门，覆盖主要知识点） */
const COURSES = [
  ['C-0001', '数据清洗实战：从脏数据到可用数据集', '数据与人工智能', 2, 8.5, 'project', 4.7, 18234, 99, ['数据清洗', '描述性统计'], '王敏'],
  ['C-0002', '描述性统计与可视化入门', '数据与人工智能', 2, 6.0, 'video', 4.5, 24510, 0, ['描述性统计'], '李泽宇'],
  ['C-0003', '假设检验：A/B 实验的统计底气', '数据与人工智能', 3, 9.0, 'mixed', 4.6, 12088, 129, ['假设检验', '概率论'], '陈思'],
  ['C-0004', '回归建模与业务解释', '数据与人工智能', 4, 11.5, 'project', 4.8, 9032, 199, ['回归建模', '最优化基础'], '赵晓峰'],
  ['C-0005', '模型评估与调优工作流', '数据与人工智能', 4, 7.5, 'text', 4.4, 6410, 0, ['模型评估'], '刘倩'],
  ['C-0006', 'Python 编程基础：变量与流程控制', '编程开发', 1, 5.0, 'video', 4.6, 38210, 0, ['变量与数据类型', '流程控制'], '孙浩然'],
  ['C-0007', '函数式思维与模块化设计', '编程开发', 2, 6.5, 'mixed', 4.5, 15870, 79, ['函数与模块'], '王敏'],
  ['C-0008', '面向对象与并发编程进阶', '编程开发', 4, 12.0, 'project', 4.3, 5210, 199, ['面向对象', '并发编程'], '李泽宇'],
  ['C-0009', '数据分析看板：从指标到故事', '产品与职场', 3, 7.0, 'project', 4.7, 9870, 99, ['数据分析看板', '需求调研'], '陈思'],
  ['C-0010', '前端工程化与组件化开发', '前端与设计', 3, 10.0, 'mixed', 4.4, 11020, 129, ['组件化开发', 'CSS 布局'], '赵晓峰'],
  ['C-0011', 'Web 安全攻防基础', '网络与安全', 3, 8.0, 'video', 4.5, 8320, 99, ['Web 攻防基础', '网络分层模型'], '刘倩'],
  ['C-0012', '概率论：从直觉到计算', '数学基础', 2, 9.5, 'text', 4.6, 14300, 0, ['概率论'], '孙浩然'],
  ['C-0013', '学术写作与商务表达', '语言学习', 3, 4.5, 'text', 4.2, 4100, 0, ['学术写作', '商务表达'], '王敏'],
  ['C-0014', '用户故事与增长实验设计', '产品与职场', 3, 6.5, 'mixed', 4.5, 7320, 79, ['用户故事', '增长实验'], '李泽宇'],
].map(([course_id, title, domain, level, duration_hours, modality, rating, enroll_count, price, kpNames, instructor]) => ({
  course_id,
  title,
  domain,
  level,
  duration_hours,
  modality,
  modality_name: MODALITY_NAMES[modality],
  rating,
  enroll_count,
  price,
  instructor,
  kp_ids: kpNames.map((name) => KP_BY_NAME[name]?.id).filter(Boolean),
  kp_names: kpNames,
  theory_ratio: 0.6,
  language: 'zh',
  tags: [domain],
  kp_coverage: kpNames.map((name, i) => ({ kp_id: KP_BY_NAME[name]?.id, weight: i === 0 ? 0.8 : 0.6 })),
}));

/** 确定性伪随机 */
const seeded = (seed) => {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
};

/** 学习者列表（顶部切换器） */
const LEARNERS = [
  ['L-0001', '学员01', 'goal_da', '数据分析师', 0.61, 8.0],
  ['L-0002', '学员02', 'goal_da', '数据分析师', 0.47, 5.0],
  ['L-0003', '学员03', 'goal_fe', '前端工程师', 0.72, 12.0],
  ['L-0004', '学员04', 'goal_fe', '前端工程师', 0.38, 3.0],
  ['L-0005', '学员05', 'goal_sec', '网络安全工程师', 0.66, 10.0],
  ['L-0006', '学员06', 'goal_da', '数据分析师', 0.83, 15.0],
  ['L-0007', '学员07', 'goal_sec', '网络安全工程师', 0.55, 6.0],
  ['L-0008', '学员08', 'goal_fe', '前端工程师', 0.29, 5.0],
].map(([learner_id, nickname, goal_id, goal_name, ability, budget]) => ({
  learner_id,
  nickname,
  goal_id,
  goal_name,
  ability,
  time_budget_hours_per_week: budget,
}));

const GOALS = {
  goal_da: {
    goal_id: 'goal_da',
    name: '数据分析师',
    target_role: '数据分析师',
    description: '面向零基础转行，覆盖数据处理、统计推断与可视化',
    target_kp_ids: ['数据清洗', '描述性统计', '假设检验', '回归建模', '数据分析看板'].map((n) => KP_BY_NAME[n].id),
    default_weeks: 12,
    required_mastery: 0.8,
  },
  goal_fe: {
    goal_id: 'goal_fe',
    name: '前端工程师',
    target_role: '前端工程师',
    description: '从前端三件套到组件化与性能优化',
    target_kp_ids: ['HTML 语义化', 'CSS 布局', 'JavaScript 基础', '组件化开发', '性能优化'].map((n) => KP_BY_NAME[n].id),
    default_weeks: 16,
    required_mastery: 0.8,
  },
  goal_sec: {
    goal_id: 'goal_sec',
    name: '网络安全工程师',
    target_role: '网络安全工程师',
    description: '网络协议基础 + Web 攻防 + 加密体系',
    target_kp_ids: ['网络分层模型', 'TCP 与 UDP', 'Web 攻防基础', '加密与证书'].map((n) => KP_BY_NAME[n].id),
    default_weeks: 14,
    required_mastery: 0.8,
  },
};

const PRESETS = [
  { key: 'balanced', name: '均衡方案', weights: { gain: 0.4, time: 0.25, smoothness: 0.2, fit: 0.15 } },
  { key: 'fast', name: '速通方案', weights: { gain: 0.3, time: 0.45, smoothness: 0.1, fit: 0.15 } },
  { key: 'thorough', name: '夯实方案', weights: { gain: 0.5, time: 0.15, smoothness: 0.2, fit: 0.15 } },
];

const PRESET_NAMES = { balanced: '均衡方案', fast: '速通方案', thorough: '夯实方案' };

/** meta */
export function mockMeta() {
  const nCourses = 28;
  return {
    stats: {
      n_knowledge_points: KPS.length,
      n_kp_edges: EDGES.length,
      n_courses: nCourses,
      n_learners: 12,
      n_goals: 3,
      n_interactions: 180,
      n_completed: 88,
      completion_rate: 0.4889,
      domains: DOMAINS,
    },
    graph: {
      n_nodes: KPS.length,
      n_edges: EDGES.length,
      n_roots: 7,
      n_leaves: 7,
      max_depth: 5,
      avg_out_degree: Number((EDGES.length / KPS.length).toFixed(3)),
      domains: DOMAINS,
    },
    domains: DOMAINS.map((name) => ({
      name,
      icon: DOMAIN_ICONS[name],
      n_kps: 5,
      n_courses: 4,
    })),
    goals: Object.values(GOALS).map((goal) => ({
      goal_id: goal.goal_id,
      name: goal.name,
      target_role: goal.target_role,
      description: goal.description,
      n_target_kps: goal.target_kp_ids.length,
      default_weeks: goal.default_weeks,
    })),
    presets: PRESETS,
    learners: LEARNERS,
    modalities: Object.entries(MODALITY_NAMES).map(([key, name]) => ({ key, name })),
    motivations: ['求职', '晋升', '兴趣', '学业', '转行'],
    data_source: 'tiny',
  };
}

const learnerOf = (learnerId) => LEARNERS.find((l) => l.learner_id === learnerId) || LEARNERS[0];

/** 依据学习者能力生成掌握度画像（确定性） */
function buildMastery(learner) {
  const goal = GOALS[learner.goal_id];
  const mastery = {};
  KPS.forEach((kp, index) => {
    const isTarget = goal.target_kp_ids.includes(kp.id);
    const noise = seeded(index + Number(learner.ability) * 100);
    if (isTarget) {
      mastery[kp.id] = Math.min(0.95, Math.max(0.12, learner.ability - 0.18 * (kp.difficulty - 2) + 0.16 * noise));
    } else if (noise > 0.62) {
      mastery[kp.id] = Math.min(0.92, Math.max(0.1, learner.ability * 0.9 + 0.1 * noise));
    }
  });
  return mastery;
}

function domainMasteryOf(mastery) {
  const result = {};
  DOMAINS.forEach((domain) => {
    const values = KPS.filter((kp) => kp.domain === domain).map((kp) => mastery[kp.id]).filter((v) => v != null);
    result[domain] = values.length ? Number((values.reduce((a, b) => a + b, 0) / values.length).toFixed(3)) : 0.24;
  });
  return result;
}

function buildProfile(learner) {
  const goal = GOALS[learner.goal_id];
  const mastery = buildMastery(learner);
  const targetKps = goal.target_kp_ids;
  const masteryNames = Object.entries(mastery)
    .map(([kpId, value]) => ({ kp_id: kpId, name: KP_BY_ID[kpId]?.name ?? kpId, value: Number(value.toFixed(3)) }))
    .sort((a, b) => b.value - a.value);
  const gaps = {};
  targetKps.forEach((kpId) => {
    const value = mastery[kpId];
    if (value == null) {
      gaps[kpId] = 0.62;
    } else if (value < 0.6) {
      gaps[kpId] = Number((0.6 - value + 0.35).toFixed(3));
    }
  });
  const gapIds = Object.keys(gaps);
  const readiness = Number(
    (targetKps.filter((id) => (mastery[id] ?? 0) >= 0.6).length / targetKps.length).toFixed(3)
  );
  const avg = masteryNames.length
    ? masteryNames.reduce((sum, item) => sum + item.value, 0) / masteryNames.length
    : 0;
  return {
    learner_id: learner.learner_id,
    mastery,
    mastery_names: masteryNames,
    domain_mastery: domainMasteryOf(mastery),
    target_kps: targetKps,
    target_kp_names: targetKps.map((id) => KP_BY_ID[id].name),
    gaps,
    gap_names: gapIds.map((id) => KP_BY_ID[id].name),
    readiness,
    style_vector: { theory: 0.58, practice: 0.42, video: 0.36, text: 0.24, project: 0.4 },
    time_budget: learner.time_budget_hours_per_week,
    total_available_hours: Number((learner.time_budget_hours_per_week * goal.default_weeks).toFixed(1)),
    mastered_count: masteryNames.filter((item) => item.value >= 0.85).length,
    gap_count: masteryNames.filter((item) => item.value < 0.6).length,
    completion_rate: Number((0.34 + 0.3 * learner.ability).toFixed(3)),
    _avg: avg,
  };
}

/** 学习者画像详情 */
export function mockLearnerDetail(learnerId) {
  const learner = learnerOf(learnerId);
  const goal = GOALS[learner.goal_id];
  const profile = buildProfile(learner);
  delete profile._avg;
  return {
    learner: {
      learner_id: learner.learner_id,
      nickname: learner.nickname,
      background: { education: '本科', major: '信息管理', work_years: 2, current_role: '运营专员' },
      goal_id: learner.goal_id,
      motivation: ['求职', '晋升', '兴趣', '学业', '转行'][Number(learner.learner_id.slice(-1)) % 5],
      deadline_weeks: goal.default_weeks,
      time_budget_hours_per_week: learner.time_budget_hours_per_week,
      learning_style: {
        theory_preference: 0.58,
        prefer_modality: 'video',
        session_len_min: 45,
        weekly_days: 4,
      },
      ability: learner.ability,
      initial_mastery: profile.mastery,
      interests: [KP_BY_ID[goal.target_kp_ids[0]].domain],
      joined_at: '2026-03-12T09:00:00',
    },
    goal: { ...goal, target_kp_ids: goal.target_kp_ids },
    profile,
    history: [
      ['C-0006', 'Python 编程基础：变量与流程控制', '编程开发', 1, true, 4.5, 5.2, 0.88, '2026-05-18T10:00:00'],
      ['C-0002', '描述性统计与可视化入门', '数据与人工智能', 1, true, 4.0, 6.4, 0.79, '2026-05-11T10:00:00'],
      ['C-0001', '数据清洗实战：从脏数据到可用数据集', '数据与人工智能', 0.72, false, null, 6.1, null, '2026-05-04T10:00:00'],
      ['C-0007', '函数式思维与模块化设计', '编程开发', 1, true, 4.5, 6.8, 0.85, '2026-04-27T10:00:00'],
      ['C-0013', '学术写作与商务表达', '语言学习', 1, true, 4.0, 4.6, 0.74, '2026-04-20T10:00:00'],
      ['C-0012', '概率论：从直觉到计算', '数学基础', 0.35, false, null, 3.4, null, '2026-04-13T10:00:00'],
    ].map(([course_id, title, domain, progress, completed, rating, hours_spent, quiz_score, timestamp]) => ({
      course_id,
      title,
      domain,
      progress,
      completed,
      rating,
      hours_spent,
      quiz_score,
      timestamp,
    })),
  };
}

const RECOMMEND_REASONS = [
  (name, value) => `填补目标缺口「${name}」（当前掌握度 ${value.toFixed(2)}）`,
  () => '先修知识已掌握 3/4，可顺利开始',
  () => '与你背景相近的 12 位学员完成率 78%',
  () => '难度梯度与当前能力匹配，跳一跳够得着',
  () => '与你偏好的「项目实战」模态一致',
  () => '完成该课可解锁后续 3 门进阶课程',
];

/** 推荐列表 */
export function mockRecommend(learnerId, k = 10) {
  const learner = learnerOf(learnerId);
  const profile = buildProfile(learner);
  const pool = COURSES.filter((c) => c.domain === '数据与人工智能' || c.domain === '数学基础' || c.domain === '产品与职场');
  const items = pool.slice(0, Math.max(3, Math.min(k, 8))).map((course, index) => {
    const score = Number((0.92 - index * 0.045).toFixed(3));
    const p = Number((0.83 - index * 0.055).toFixed(3));
    const gapName = profile.gap_names[index % Math.max(1, profile.gap_names.length)] || course.kp_names[0];
    const gapValue = 0.24 + index * 0.03;
    return {
      rank: index + 1,
      course_id: course.course_id,
      title: course.title,
      domain: course.domain,
      level: course.level,
      duration_hours: course.duration_hours,
      modality: course.modality,
      modality_name: course.modality_name,
      rating: course.rating,
      enroll_count: course.enroll_count,
      price: course.price,
      kp_names: course.kp_names,
      score,
      components: {
        content: Number((0.94 - index * 0.03).toFixed(3)),
        collaborative: Number((0.68 - index * 0.02).toFixed(3)),
        quality: Number((0.88 - index * 0.015).toFixed(3)),
        readiness: Number((0.82 - index * 0.035).toFixed(3)),
        social: Number((0.61 - index * 0.02).toFixed(3)),
      },
      reasons: [
        RECOMMEND_REASONS[0](gapName, gapValue),
        RECOMMEND_REASONS[1](),
        RECOMMEND_REASONS[2 + (index % 4)](),
      ],
      expected_gain: Number((0.21 - index * 0.021).toFixed(3)),
      p_completion: p,
      risk_level: p >= 0.7 ? '低' : p >= 0.4 ? '中' : '高',
      readiness: Number((0.79 - index * 0.02).toFixed(3)),
      style_fit: Number((0.86 - index * 0.025).toFixed(3)),
      time_fit: Number((0.91 - index * 0.03).toFixed(3)),
    };
  });
  return { learner_id: learner.learner_id, k, count: items.length, items };
}

const PATH_STEPS = [
  ['C-0006', 'Python 编程基础：变量与流程控制', '编程开发', 1, 5.0, 'video', ['变量与数据类型', '流程控制'], 0.19],
  ['C-0007', '函数式思维与模块化设计', '编程开发', 2, 6.5, 'mixed', ['函数与模块'], 0.17],
  ['C-0001', '数据清洗实战：从脏数据到可用数据集', '数据与人工智能', 2, 8.5, 'project', ['数据清洗'], 0.16],
  ['C-0002', '描述性统计与可视化入门', '数据与人工智能', 2, 6.0, 'video', ['描述性统计'], 0.15],
  ['C-0012', '概率论：从直觉到计算', '数学基础', 2, 9.5, 'text', ['概率论'], 0.12],
  ['C-0003', '假设检验：A/B 实验的统计底气', '数据与人工智能', 3, 9.0, 'mixed', ['假设检验'], 0.14],
  ['C-0004', '回归建模与业务解释', '数据与人工智能', 4, 11.5, 'project', ['回归建模'], 0.16],
  ['C-0009', '数据分析看板：从指标到故事', '产品与职场', 3, 7.0, 'project', ['数据分析看板'], 0.11],
];

function buildSteps(scale = 1) {
  let cumulative = 0;
  let week = 1;
  return PATH_STEPS.slice(0, Math.max(4, Math.round(PATH_STEPS.length * scale))).map((step, index) => {
    const [course_id, title, domain, level, baseHours, modality, kp_names, gain] = step;
    const hours = Number((baseHours * (scale < 0.8 ? 0.82 : 1)).toFixed(1));
    cumulative = Number((cumulative + hours).toFixed(1));
    week = Math.max(1, Math.ceil(cumulative / 8));
    return {
      order: index + 1,
      course_id,
      title,
      domain,
      level,
      hours,
      cumulative_hours: cumulative,
      week,
      kp_names,
      expected_gain: gain,
      modality,
    };
  });
}

function summarize(learner, preset, steps) {
  const totalHours = Number(steps.reduce((sum, s) => sum + s.hours, 0).toFixed(1));
  const totalGain = Number(steps.reduce((sum, s) => sum + s.expected_gain, 0).toFixed(3));
  const weeks = steps[steps.length - 1]?.week ?? 1;
  const objectives = {
    balanced: { gain: 0.71, time: 0.66, smoothness: 0.82, fit: 0.75 },
    fast: { gain: 0.58, time: 0.88, smoothness: 0.61, fit: 0.7 },
    thorough: { gain: 0.86, time: 0.42, smoothness: 0.9, fit: 0.81 },
  }[preset];
  const feasibility = Math.min(1, Number((learner.time_budget_hours_per_week * weeks / Math.max(1, totalHours)).toFixed(3)));
  return { totalHours, totalGain, weeks, objectives, feasibility };
}

/** 目标知识点覆盖（示例数据：按目标点数取前 N 个视为已覆盖） */
function coverageOf(learner, coveredCount) {
  const goal = GOALS[learner.goal_id];
  const all = goal.target_kp_ids;
  const coveredIds = all.slice(0, Math.max(0, Math.min(all.length, coveredCount)));
  const uncoveredIds = all.slice(coveredIds.length);
  const ratio = all.length ? Number((coveredIds.length / all.length).toFixed(3)) : 0;
  const budget = Number((learner.time_budget_hours_per_week * goal.default_weeks).toFixed(1));
  return {
    coverage_ratio: ratio,
    covered_target_kp_ids: coveredIds,
    covered_target_kp_names: coveredIds.map((id) => KP_BY_ID[id]?.name ?? id),
    uncovered_target_kp_ids: uncoveredIds,
    uncovered_target_kp_names: uncoveredIds.map((id) => KP_BY_ID[id]?.name ?? id),
    budget_hours: budget,
  };
}

/** 学习路径 */
export function mockPath(learnerId, preset = 'balanced') {
  const learner = learnerOf(learnerId);
  const goal = GOALS[learner.goal_id];
  const scale = preset === 'fast' ? 0.85 : preset === 'thorough' ? 1 : 0.95;
  const steps = buildSteps(scale);
  const summary = summarize(learner, preset, steps);
  const coveredCount = preset === 'fast' ? 3 : preset === 'thorough' ? 5 : 4;
  const cover = coverageOf(learner, coveredCount);
  const weeklyPlan = [];
  steps.forEach((step) => {
    const bucket = weeklyPlan.find((row) => row.week === step.week);
    if (bucket) {
      bucket.hours = Number((bucket.hours + step.hours).toFixed(1));
      bucket.course_ids.push(step.course_id);
    } else {
      weeklyPlan.push({ week: step.week, hours: step.hours, course_ids: [step.course_id] });
    }
  });
  const milestones = steps
    .filter((step) => step.order % 2 === 0)
    .slice(0, 4)
    .map((step) => ({
      week: step.week,
      label: `掌握 ${step.kp_names.join('、')}`,
      kp_names: step.kp_names,
    }));
  const alternatives = Object.keys(PRESET_NAMES)
    .filter((key) => key !== preset)
    .map((key) => {
      const altSteps = buildSteps(key === 'fast' ? 0.85 : key === 'thorough' ? 1 : 0.95);
      const alt = summarize(learner, key, altSteps);
      const altCover = coverageOf(learner, key === 'fast' ? 3 : key === 'thorough' ? 5 : 4);
      return {
        preset: key,
        preset_name: PRESET_NAMES[key],
        total_hours: alt.totalHours,
        weeks: alt.weeks,
        total_gain: alt.totalGain,
        feasibility: alt.feasibility,
        coverage_ratio: altCover.coverage_ratio,
        objectives: alt.objectives,
      };
    });

  return {
    learner_id: learner.learner_id,
    preset,
    preset_name: PRESET_NAMES[preset],
    goal: { goal_id: goal.goal_id, name: goal.name },
    total_hours: summary.totalHours,
    weeks: summary.weeks,
    total_gain: summary.totalGain,
    feasibility: summary.feasibility,
    objectives: summary.objectives,
    budget_hours: cover.budget_hours,
    over_budget: summary.totalHours > cover.budget_hours,
    coverage_ratio: cover.coverage_ratio,
    covered_target_kp_ids: cover.covered_target_kp_ids,
    covered_target_kp_names: cover.covered_target_kp_names,
    uncovered_target_kp_ids: cover.uncovered_target_kp_ids,
    uncovered_target_kp_names: cover.uncovered_target_kp_names,
    coverage: {
      target_gap_ratio: Number((1 - cover.coverage_ratio).toFixed(3)),
      gap_ratio: 0.12,
      budget_utilisation: Number((summary.totalHours / Math.max(1, cover.budget_hours)).toFixed(3)),
      weeks_needed: summary.weeks,
      deadline_weeks: goal.default_weeks,
      avg_weekly_load: Number((summary.totalHours / Math.max(1, summary.weeks)).toFixed(2)),
      hours_shortfall: Math.max(0, Number((summary.totalHours - cover.budget_hours).toFixed(1))),
    },
    steps,
    weekly_plan: weeklyPlan,
    milestones,
    alternatives,
  };
}

/** 帕累托前沿 */
export function mockPareto(learnerId, preset = 'balanced') {
  const learner = learnerOf(learnerId);
  const points = [];
  for (let i = 0; i < 9; i += 1) {
    const hours = Number((26 + i * 5.2).toFixed(1));
    const gain = Number((1.35 + Math.log(hours / 24) * 1.55 - 0.012 * i).toFixed(3));
    points.push({
      id: `p${i + 1}`,
      hours,
      gain,
      smoothness: Number((0.52 + 0.045 * i).toFixed(3)),
      fit: Number((0.62 + 0.03 * (i % 4)).toFixed(3)),
      n_courses: 5 + Math.round(i * 0.8),
      is_selected: false,
      preset: null,
      weeks: Math.ceil(hours / 8),
    });
  }
  Object.keys(PRESET_NAMES).forEach((key) => {
    const alt = mockPath(learnerId, key);
    points.push({
      id: `preset-${key}`,
      hours: alt.total_hours,
      gain: alt.total_gain,
      smoothness: alt.objectives.smoothness,
      fit: alt.objectives.fit,
      n_courses: alt.steps.length,
      is_selected: key === preset,
      preset: key,
      weeks: alt.weeks,
    });
  });
  return { learner_id: learner.learner_id, x: 'hours', y: 'gain', points };
}

/** 学习分析仪表板 */
export function mockAnalytics(learnerId) {
  const learner = learnerOf(learnerId);
  const goal = GOALS[learner.goal_id];
  const profile = buildProfile(learner);
  const path = mockPath(learnerId, 'balanced');
  const masteryValues = profile.mastery_names.map((item) => item.value);
  const avgMastery = masteryValues.reduce((a, b) => a + b, 0) / Math.max(1, masteryValues.length);

  const masteryCurve = Array.from({ length: 9 }, (_, week) => {
    const progress = 1 - Math.exp(-week / 4.2);
    return {
      week,
      avg_mastery: Number((0.26 + (avgMastery + 0.28 - 0.26) * progress * 0.92).toFixed(3)),
      mastered: Math.round(progress * profile.target_kps.length * 0.9),
    };
  });

  const gapTable = Object.entries(profile.gaps)
    .slice(0, 6)
    .map(([kpId, value]) => ({
      kp_id: kpId,
      name: KP_BY_ID[kpId].name,
      mastery: Number(((profile.mastery[kpId] ?? 0.18)).toFixed(3)),
      difficulty: KP_BY_ID[kpId].difficulty,
      domain: KP_BY_ID[kpId].domain,
      in_path: path.steps.some((step) => step.kp_names.includes(KP_BY_ID[kpId].name)),
    }));

  const forecast = path.steps.slice(0, 6).map((step, index) => {
    const p = Number((0.86 - index * 0.06).toFixed(3));
    return {
      order: step.order,
      course_id: step.course_id,
      course_title: step.title,
      p_completion: p,
      risk_level: p >= 0.7 ? '低' : p >= 0.4 ? '中' : '高',
    };
  });

  const modalityHours = {};
  path.steps.forEach((step) => {
    modalityHours[step.modality] = Number(((modalityHours[step.modality] ?? 0) + step.hours).toFixed(1));
  });

  return {
    learner_id: learner.learner_id,
    kpi: {
      overall_progress: profile.completion_rate,
      mastered_kps: profile.mastered_count,
      total_target_kps: goal.target_kp_ids.length,
      avg_mastery: Number(avgMastery.toFixed(3)),
      readiness: profile.readiness,
      planned_hours: path.total_hours,
      available_hours: profile.total_available_hours,
      risk_courses: forecast.filter((row) => row.risk_level !== '低').length,
      learning_streak_weeks: 5,
    },
    mastery_curve: masteryCurve,
    domain_radar: DOMAINS.map((domain) => ({
      domain,
      mastery: profile.domain_mastery[domain] ?? 0.3,
      target: 0.8,
    })),
    difficulty_distribution: [1, 2, 3, 4, 5].map((level) => ({
      level,
      count: path.steps.filter((step) => step.level === level).length,
    })),
    modality_mix: Object.entries(modalityHours).map(([modality, hours]) => ({
      modality,
      name: MODALITY_NAMES[modality] ?? modality,
      hours,
    })),
    time_allocation: Array.from({ length: Math.max(path.weeks, 6) }, (_, index) => {
      const week = index + 1;
      const row = path.weekly_plan.find((item) => item.week === week);
      return {
        week,
        hours: row ? row.hours : 0,
        planned: learner.time_budget_hours_per_week,
      };
    }),
    completion_forecast: forecast,
    gap_table: gapTable,
    history_activity: Array.from({ length: Math.max(path.weeks, 6) }, (_, index) => ({
      week: index + 1,
      hours: Number((2.4 + 3.1 * seeded(index + 3) + index * 0.35).toFixed(1)),
      completed: index % 2,
    })),
  };
}

/** 社交学习 */
export function mockSocial(learnerId, k = 6) {
  const learner = learnerOf(learnerId);
  const profile = buildProfile(learner);
  const peerPool = LEARNERS.filter((item) => item.learner_id !== learner.learner_id);
  const peers = peerPool.slice(0, Math.min(k, 5)).map((peer, index) => {
    const peerProfile = buildProfile(peer);
    const aheadIn = peerProfile.mastery_names
      .filter((item) => item.value > (profile.mastery[item.kp_id] ?? 0) + 0.12)
      .slice(0, 3)
      .map((item) => item.name);
    return {
      learner_id: peer.learner_id,
      nickname: peer.nickname,
      ability: peer.ability,
      goal_name: peer.goal_name,
      similarity: Number((0.89 - index * 0.047).toFixed(3)),
      shared_goal_kps: profile.target_kp_names.slice(0, 3),
      shared_courses: ['数据清洗实战', '描述性统计入门'].slice(0, 2 - (index % 2)),
      ahead_in: aheadIn.length ? aheadIn : ['回归建模'],
      completed_count: 4 + (index % 4),
    };
  });

  const groups = [
    {
      group_id: 'G-01',
      kp_names: ['假设检验', '回归建模'],
      members: [
        { learner_id: learner.learner_id, nickname: learner.nickname, ability: learner.ability },
        { learner_id: 'L-0006', nickname: '学员06', ability: 0.83 },
        { learner_id: 'L-0002', nickname: '学员02', ability: 0.47 },
      ],
      avg_completion: 0.68,
      synergy: 0.81,
    },
    {
      group_id: 'G-02',
      kp_names: ['数据清洗', '描述性统计'],
      members: [
        { learner_id: learner.learner_id, nickname: learner.nickname, ability: learner.ability },
        { learner_id: 'L-0004', nickname: '学员04', ability: 0.38 },
      ],
      avg_completion: 0.55,
      synergy: 0.72,
    },
    {
      group_id: 'G-03',
      kp_names: ['数据分析看板'],
      members: [
        { learner_id: 'L-0003', nickname: '学员03', ability: 0.72 },
        { learner_id: 'L-0005', nickname: '学员05', ability: 0.66 },
      ],
      avg_completion: 0.74,
      synergy: 0.67,
    },
  ];

  const rates = peers.map((peer, index) => Number((0.52 + index * 0.04).toFixed(3)));
  return {
    learner_id: learner.learner_id,
    peers,
    groups,
    cohort_summary: {
      n_peers: peers.length,
      avg_similarity: Number(
        (peers.reduce((sum, peer) => sum + peer.similarity, 0) / Math.max(1, peers.length)).toFixed(3)
      ),
      cohort_completion_rate: Number(
        (rates.reduce((a, b) => a + b, 0) / Math.max(1, rates.length)).toFixed(3)
      ),
      your_completion_rate: profile.completion_rate,
    },
  };
}

/** 系统评测 */
export function mockEvaluation() {
  return {
    generated_at: '2026-10-29T18:00:00',
    recommenders: [
      {
        name: '热度基线 Popularity',
        precision_at_k: 0.061,
        recall_at_k: 0.184,
        ndcg_at_k: 0.212,
        map: 0.131,
        coverage: 0.041,
        diversity: 0.32,
        novelty: 0.18,
        hit_rate: 0.31,
      },
      {
        name: '内容推荐 Content',
        precision_at_k: 0.118,
        recall_at_k: 0.302,
        ndcg_at_k: 0.361,
        map: 0.238,
        coverage: 0.196,
        diversity: 0.44,
        novelty: 0.27,
        hit_rate: 0.52,
      },
      {
        name: '协同过滤 CF-SVD',
        precision_at_k: 0.142,
        recall_at_k: 0.347,
        ndcg_at_k: 0.402,
        map: 0.271,
        coverage: 0.143,
        diversity: 0.38,
        novelty: 0.24,
        hit_rate: 0.58,
      },
      {
        name: '混合推荐 Hybrid（本系统）',
        precision_at_k: 0.187,
        recall_at_k: 0.421,
        ndcg_at_k: 0.489,
        map: 0.336,
        coverage: 0.284,
        diversity: 0.57,
        novelty: 0.39,
        hit_rate: 0.71,
      },
    ],
    effect_model: {
      task: 'course_completion',
      positive_rate: 0.49,
      models: [
        { name: '逻辑回归', auc: 0.79, accuracy: 0.72, f1: 0.74, cv_mean: 0.785, cv_std: 0.011 },
        { name: '梯度提升树', auc: 0.83, accuracy: 0.76, f1: 0.78, cv_mean: 0.826, cv_std: 0.008 },
      ],
      best: '梯度提升树',
      features: [
        { name: 'learner_ability', importance: 0.21 },
        { name: 'prereq_readiness', importance: 0.18 },
        { name: 'course_level_gap', importance: 0.15 },
        { name: 'duration_hours', importance: 0.12 },
        { name: 'time_budget_ratio', importance: 0.1 },
        { name: 'style_fit', importance: 0.08 },
      ],
    },
    gain_model: {
      task: 'mastery_gain_regression',
      best: '梯度提升树',
      models: [
        { name: '线性回归', mae: 0.062, rmse: 0.081, r2: 0.54 },
        { name: '梯度提升树', mae: 0.041, rmse: 0.058, r2: 0.71 },
      ],
    },
    ablation: [
      { variant: '完整模型（本系统）', ndcg_at_k: 0.489, delta: 0 },
      { variant: '去掉知识图谱先修约束', ndcg_at_k: 0.403, delta: -0.086 },
      { variant: '去掉多目标优化', ndcg_at_k: 0.451, delta: -0.038 },
      { variant: '去掉效果预测重排', ndcg_at_k: 0.466, delta: -0.023 },
      { variant: '去掉多样性重排序 MMR', ndcg_at_k: 0.472, delta: -0.017 },
    ],
    notes: [
      '离线评测采用留一法划分：每位学习者保留 20% 的正反馈作为测试集，其余用于训练。',
      '覆盖率 Coverage 统计推荐结果覆盖的课程占课程库的比例，反映长尾挖掘能力。',
      '多样性 Diversity 为推荐列表内两两课程的知识点 Jaccard 距离均值。',
      '新颖度 Novelty 基于课程历史曝光量的自信息，越高说明越能推出冷门课程。',
    ],
  };
}

/** 知识图谱（支持领域过滤与掌握度染色） */
export function mockGraph({ domain = null, learner_id = null } = {}) {
  const learner = learner_id ? learnerOf(learner_id) : null;
  const profile = learner ? buildProfile(learner) : null;
  const goal = learner ? GOALS[learner.goal_id] : null;
  const nodesPool = KPS.filter((kp) => !domain || kp.domain === domain);
  const keep = new Set(nodesPool.map((kp) => kp.id));
  const categories = [...new Set(nodesPool.map((kp) => kp.domain))];
  const nodes = nodesPool.map((kp) => ({
    id: kp.id,
    name: kp.name,
    category: categories.indexOf(kp.domain),
    domain: kp.domain,
    difficulty: kp.difficulty,
    depth: kp.depth,
    mastery: profile ? Number(((profile.mastery[kp.id] ?? 0)).toFixed(3)) : null,
    symbolSize: 12 + 4 * kp.difficulty,
    highlight: Boolean(goal && goal.target_kp_ids.includes(kp.id)),
  }));
  const links = EDGES.filter((edge) => keep.has(edge.source) && keep.has(edge.target));
  const pathKps = mockPath(learner ? learner.learner_id : 'L-0001', 'balanced')
    .steps.flatMap((step) => step.kp_names)
    .map((name) => KP_BY_NAME[name]?.id)
    .filter(Boolean);
  return {
    nodes,
    links,
    categories,
    focus: {
      target_kps: goal ? goal.target_kp_ids : [],
      path_kps: [...new Set(pathKps)],
    },
  };
}

/** 学习反馈（示例：用于前端演示提交流程时展示） */
export function mockFeedback(learnerId, courseId) {
  const learner = learnerOf(learnerId);
  const profile = buildProfile(learner);
  return {
    learner_id: learner.learner_id,
    course_id: courseId,
    mastery_updates: profile.mastery,
    delta: Object.fromEntries(Object.keys(profile.mastery).slice(0, 3).map((id) => [id, 0.12])),
    actions: ['掌握度提升显著，已解锁后续课程 3 门', '路径已自动重排', '跳过 1 门已掌握课程'],
    replanned: true,
    reason: '结课测验 0.85 高于阈值 0.60，按 BKT 更新掌握度并跳过冗余课程',
    profile,
    path: mockPath(learner.learner_id, 'balanced'),
  };
}

export const MOCK = {
  meta: mockMeta,
  learnerDetail: mockLearnerDetail,
  recommend: mockRecommend,
  path: mockPath,
  pareto: mockPareto,
  analytics: mockAnalytics,
  social: mockSocial,
  evaluation: mockEvaluation,
  graph: mockGraph,
};
