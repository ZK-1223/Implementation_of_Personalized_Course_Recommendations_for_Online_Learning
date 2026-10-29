/**
 * REST 客户端：封装 fetch、超时、错误语义。
 *
 * 所有接口严格对应 docs/API.md。任何非 2xx 都抛 ApiError（携带 status 与后端 detail），
 * 视图层据 status 决定是「展示错误」还是「回退到离线示例数据」。
 */

const TIMEOUT_MS = 20000;

/**
 * 各接口的超时预算（毫秒）。
 *
 * 首次调用部分接口会触发重计算（效果预测模型训练、NSGA-II 路径优化、
 * 离线评测的 CF-SVD 训练与交叉验证），因此单独放宽；结果在服务端有缓存，
 * 后续请求会恢复到百毫秒级。
 */
const TIMEOUTS = {
  '/api/evaluation': 180000,
  '/api/path': 90000,
  '/api/analytics': 90000,
  '/api/recommend': 90000,
  '/api/social': 60000,
  '/api/learners': 60000,
};

const timeoutFor = (path) => {
  const hit = Object.keys(TIMEOUTS).find((prefix) => path.startsWith(prefix));
  return hit ? TIMEOUTS[hit] : TIMEOUT_MS;
};

export class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

async function request(path, { method = 'GET', body = null, timeout = timeoutFor(path) } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      cache: 'no-store',
    });
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { detail: text.slice(0, 300) };
      }
    }
    if (!response.ok) {
      const raw = payload?.detail;
      const detail =
        typeof raw === 'string'
          ? raw
          : Array.isArray(raw)
            ? raw.map((item) => item?.msg || String(item)).join('；')
            : raw
              ? JSON.stringify(raw)
              : `请求失败（HTTP ${response.status}）`;
      throw new ApiError(detail, response.status, payload);
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error?.name === 'AbortError') {
      throw new ApiError('请求超时，后端可能正在计算（首次加载会训练模型）', 0);
    }
    throw new ApiError('无法连接后端服务', 0, error);
  } finally {
    clearTimeout(timer);
  }
}

const qs = (params = {}) => {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') search.append(key, value);
  });
  const text = search.toString();
  return text ? `?${text}` : '';
};

export const api = {
  health: () => request('/api/health'),
  meta: () => request('/api/meta'),
  courses: (params) => request(`/api/courses${qs(params)}`),
  learners: (params) => request(`/api/learners${qs(params)}`),
  learner: (id) => request(`/api/learners/${encodeURIComponent(id)}`),
  recommend: (id, k = 10) => request(`/api/recommend/${encodeURIComponent(id)}${qs({ k })}`),
  path: (id, preset = 'balanced') =>
    request(`/api/path/${encodeURIComponent(id)}${qs({ preset })}`),
  pareto: (id, preset = 'balanced') =>
    request(`/api/path/${encodeURIComponent(id)}/pareto${qs({ preset })}`),
  social: (id, k = 6) => request(`/api/social/${encodeURIComponent(id)}${qs({ k })}`),
  analytics: (id, preset = 'balanced') =>
    request(`/api/analytics/${encodeURIComponent(id)}${qs({ preset })}`),
  evaluation: () => request('/api/evaluation'),
  graph: (params) => request(`/api/knowledge-graph${qs(params)}`),
  feedback: (payload) => request('/api/feedback', { method: 'POST', body: payload }),
};

/** 把错误翻译成给用户看的中文说明 */
export function describeError(error) {
  if (!(error instanceof ApiError)) return '未知错误';
  if (error.status === 503) return '该接口依赖的算法模块尚未就绪';
  if (error.status === 404) return `资源不存在：${error.message}`;
  if (error.status === 422) return `请求参数不合法：${error.message}`;
  if (error.status === 0) return error.message;
  return error.message;
}

/** 是否为「模块未就绪」类错误（可用示例数据兜底） */
export function isNotReady(error) {
  return error instanceof ApiError && (error.status === 503 || error.status === 0);
}
