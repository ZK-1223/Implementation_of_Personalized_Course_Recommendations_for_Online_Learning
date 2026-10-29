/**
 * ECharts 封装：统一深色主题、实例登记、销毁与自适应。
 *
 * 页面每次切换视图都会先 disposeCharts()，避免 canvas 泄漏；
 * 容器尺寸变化（响应式、侧栏伸缩）通过 ResizeObserver 与 window.resize 双保险自适应。
 */

/* global echarts */

export const PALETTE = [
  '#2f6bff',
  '#00c48c',
  '#ffb020',
  '#8b5cf6',
  '#22d3ee',
  '#ff5b6e',
  '#4f8bff',
  '#7ff0c8',
];

const AXIS_LINE = 'rgba(148, 172, 224, 0.22)';
const SPLIT_LINE = 'rgba(148, 172, 224, 0.10)';
const TEXT_MUTED = '#8494b5';
const TEXT = '#e9effc';

let themeReady = false;

function registerTheme() {
  if (themeReady) return;
  themeReady = true;
  echarts.registerTheme('learnpath', {
    color: PALETTE,
    backgroundColor: 'transparent',
    textStyle: {
      fontFamily: 'system-ui, "Microsoft YaHei", "PingFang SC", sans-serif',
      color: '#b9c6e2',
      fontSize: 12,
    },
    title: {
      textStyle: { color: TEXT, fontWeight: 600, fontSize: 13 },
      subtextStyle: { color: TEXT_MUTED, fontSize: 11 },
    },
    legend: {
      textStyle: { color: TEXT_MUTED, fontSize: 11 },
      itemWidth: 10,
      itemHeight: 10,
      icon: 'roundRect',
      itemGap: 14,
    },
    tooltip: {
      backgroundColor: 'rgba(13, 20, 36, 0.96)',
      borderColor: 'rgba(148, 172, 224, 0.26)',
      borderWidth: 1,
      padding: [9, 12],
      textStyle: { color: TEXT, fontSize: 12 },
      extraCssText: 'border-radius:10px;box-shadow:0 12px 30px rgba(3,8,20,.5);backdrop-filter:blur(6px);',
      axisPointer: {
        lineStyle: { color: 'rgba(148, 172, 224, 0.3)' },
        crossStyle: { color: 'rgba(148, 172, 224, 0.3)' },
        shadowStyle: { color: 'rgba(47, 107, 255, 0.08)' },
      },
    },
    categoryAxis: {
      axisLine: { lineStyle: { color: AXIS_LINE } },
      axisTick: { show: false },
      axisLabel: { color: TEXT_MUTED, fontSize: 11 },
      splitLine: { show: false },
    },
    valueAxis: {
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: TEXT_MUTED, fontSize: 11 },
      splitLine: { show: true, lineStyle: { color: SPLIT_LINE, type: 'dashed' } },
    },
    radar: {
      axisName: { color: TEXT_MUTED, fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(148, 172, 224, 0.16)' } },
      splitArea: { areaStyle: { color: ['rgba(148, 172, 224, 0.03)', 'rgba(148, 172, 224, 0.06)'] } },
      axisLine: { lineStyle: { color: 'rgba(148, 172, 224, 0.16)' } },
    },
    graph: {
      color: PALETTE,
      lineStyle: { color: 'rgba(148, 172, 224, 0.28)', width: 1, curveness: 0.12 },
      label: { color: TEXT },
    },
    line: { symbolSize: 6, smooth: true },
    bar: { barMaxWidth: 26, itemStyle: { borderRadius: [4, 4, 0, 0] } },
    pie: { itemStyle: { borderColor: 'rgba(9, 14, 26, 0.9)', borderWidth: 2 } },
  });
}

const registry = [];

/** 初始化图表并登记（复用容器时先销毁旧实例） */
export function chart(container, option) {
  if (!container) return null;
  registerTheme();
  const existing = echarts.getInstanceByDom(container);
  if (existing) existing.dispose();
  const instance = echarts.init(container, 'learnpath', { renderer: 'canvas' });
  instance.setOption(option, true);
  const entry = { instance, container };
  registry.push(entry);

  if (typeof ResizeObserver === 'function') {
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(container);
    entry.observer = observer;
  }
  return instance;
}

/** 按 id 初始化 */
export function chartById(id, option) {
  return chart(document.getElementById(id), option);
}

/** 销毁全部图表（视图切换 / 重渲染前调用） */
export function disposeCharts() {
  while (registry.length) {
    const entry = registry.pop();
    try {
      entry.observer?.disconnect();
      entry.instance.dispose();
    } catch {
      /* 图表已销毁 */
    }
  }
}

let resizeBound = false;

/** 绑定窗口 resize（仅一次） */
export function bindGlobalResize() {
  if (resizeBound) return;
  resizeBound = true;
  let timer = null;
  window.addEventListener('resize', () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      registry.forEach((entry) => entry.instance.resize());
    }, 140);
  });
}

/** 渐变填充（折线/柱状） */
export function gradient(color, from = 0.34, to = 0.02) {
  return {
    type: 'linear',
    x: 0,
    y: 0,
    x2: 0,
    y2: 1,
    colorStops: [
      { offset: 0, color: hexToRgba(color, from) },
      { offset: 1, color: hexToRgba(color, to) },
    ],
  };
}

function hexToRgba(hex, alpha) {
  if (!hex.startsWith('#')) return hex;
  const value = hex.slice(1);
  const full = value.length === 3 ? value.split('').map((c) => c + c).join('') : value;
  const num = parseInt(full, 16);
  const r = (num >> 16) & 255;
  const g = (num >> 8) & 255;
  const b = num & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** 通用 tooltip 数值格式化 */
export const fmt3 = (value) => (typeof value === 'number' ? value.toFixed(3) : value);
export const fmtPct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
