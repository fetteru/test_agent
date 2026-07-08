/**
 * Markdown 渲染⼯具
 * ⽤于 Day 11 智能体对话中 AI 回复的 Markdown 渲染
 */

import MarkdownIt from 'markdown-it'
// 创建 markdown-it 实例，启⽤ HTML ⽀持
const md = new MarkdownIt({
html: false, // 禁⽤ HTML 标签（安全考虑）
linkify: true, // ⾃动将 URL 转为链接
typographer: true, // 启⽤排版优化（如引号替换）
breaks: true, // 将 \n 转为 <br>
})
/**
 * 将 Markdown ⽂本渲染为 HTML
 * @param {string} text - Markdown ⽂本
 * @returns {string} 渲染后的 HTML 字符串
 */
export function renderMarkdown(text) {
if (!text) return ''
return md.render(text)
}
export default md