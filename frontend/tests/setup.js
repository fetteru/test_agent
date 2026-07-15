/**
 * Vitest 全局 setup
 * 在每个测试⽂件执⾏前⾃动运⾏
 */
import { vi } from "vitest";

// ── 模拟 Element Plus 的 ElMessage（避免测试中弹出消息框）
vi.mock("element-plus", async () => {
  const actual = await vi.importActual("element-plus");
  return {
    ...actual,
    ElMessage: {
      success: vi.fn(),
      error: vi.fn(),
      warning: vi.fn(),
      info: vi.fn(),
    },
    ElMessageBox: {
      confirm: vi.fn().mockResolvedValue("confirm"),
    },
  };
});

// ── 模拟 Element Plus 图标组件 ────────────────────────
vi.mock("@element-plus/icons-vue", () => ({
  ArrowDown: { name: "ArrowDown", render: vi.fn() },
  User: { name: "User", render: vi.fn() },
  SwitchButton: { name: "SwitchButton", render: vi.fn() },
}));

// ── 模拟 vue-router ───────────────────────────────────
vi.mock("vue-router", async () => {
  const actual = await vi.importActual("vue-router");
  return {
    ...actual,
    useRouter: vi.fn().mockReturnValue({
      push: vi.fn(),
      replace: vi.fn(),
    }),
    useRoute: vi.fn().mockReturnValue({
      path: "/",
      params: {},
      query: {},
    }),
  };
});

// ── 模拟 SVG 文件（解决 vite-svg-loader 在测试环境报错）──
vi.mock("/favicon.svg", () => ({ default: "mocked-svg" }));
vi.mock("/icons.svg", () => ({ default: "mocked-icons-svg" }));
