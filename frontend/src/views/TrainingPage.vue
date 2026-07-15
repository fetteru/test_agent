<template>
  <div class="training-page">
    <div class="page-header">
      <h2>模型训练与监控</h2>
      <el-button type="primary" @click="showCreateDialog = true">
        <el-icon><Plus /></el-icon>新建训练任务
      </el-button>
    </div>

    <!-- 训练任务列表 -->
    <el-card class="task-list-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>训练任务列表</span>
          <el-button text @click="fetchTasks">
            <el-icon><Refresh /></el-icon>刷新
          </el-button>
        </div>
      </template>

      <el-table :data="taskList" stripe style="width: 100%" v-loading="loadingTasks">
        <el-table-column prop="task_uuid" label="任务 ID" width="100" />
        <el-table-column prop="model_name" label="模型" width="110" />
        <el-table-column prop="device" label="设备" width="80" />
        <el-table-column label="进度" width="180">
          <template #default="{ row }">
            <el-progress
              :percentage="row.progress"
              :status="row.status === 'completed' ? 'success' : row.status === 'failed' ? 'exception' : ''"
              :stroke-width="16"
            />
          </template>
        </el-table-column>
        <el-table-column label="Epoch" width="100">
          <template #default="{ row }">
            {{ row.current_epoch }}/{{ row.epochs }}
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <el-tag :type="statusType(row.status)" size="small">
              {{ statusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="170" />
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="primary" text @click="selectTask(row)">监控</el-button>
            <el-button v-if="row.status === 'running'" size="small" type="danger" text @click="stopTask(row.id)">停止</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- 训练监控面板 -->
    <el-card v-if="selectedTask" class="monitor-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>
            训练监控 — 任务 {{ selectedTask.task_uuid }}
            <el-tag :type="statusType(selectedTask.status)" size="small" style="margin-left: 8px">
              {{ statusText(selectedTask.status) }}
            </el-tag>
          </span>
          <div class="monitor-info">
            <span>模型: {{ selectedTask.model_name }}</span>
            <span>设备: {{ selectedTask.device }}</span>
            <span>Epoch: {{ selectedTask.current_epoch }}/{{ selectedTask.epochs }}</span>
          </div>
        </div>
      </template>

      <el-row :gutter="16" class="metric-cards">
        <el-col :span="4" v-for="item in metricCards" :key="item.label">
          <el-card shadow="hover" class="metric-item">
            <div class="metric-value">{{ item.value }}</div>
            <div class="metric-label">{{ item.label }}</div>
          </el-card>
        </el-col>
      </el-row>

      <el-row :gutter="16" style="margin-top: 16px">
        <el-col :span="12">
          <div ref="lossChartRef" style="height: 350px"></div>
        </el-col>
        <el-col :span="12">
          <div ref="mapChartRef" style="height: 350px"></div>
        </el-col>
      </el-row>
    </el-card>

    <!-- ── 模型操作栏 ── -->
    <el-card v-if="selectedTask && selectedTask.status === 'completed'" class="action-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>模型操作</span>
        </div>
      </template>

      <el-space wrap>
        <el-button type="primary" @click="validateModel" :loading="validating">
          评估模型
        </el-button>
        <el-button type="success" @click="showExportDialog = true">
          导出模型
        </el-button>
        <el-button @click="downloadModel">
          下载权重
        </el-button>
        <el-button type="warning" @click="showPredictDialog = true">
          测试验证
        </el-button>
      </el-space>
    </el-card>

    <!-- ── 评估报告面板 ── -->
    <el-card v-if="evalReport" class="eval-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span>
            评估报告
            <el-tag size="small" style="margin-left: 8px">
              {{ evalReport.split === 'val' ? '验证集' : '测试集' }}
            </el-tag>
          </span>
        </div>
      </template>

      <!-- 整体指标 -->
      <el-row :gutter="16" class="metric-cards">
        <el-col :span="6" v-for="item in evalMetricCards" :key="item.label">
          <el-card shadow="hover" class="metric-item">
            <div class="metric-value" :style="{ color: item.color }">
              {{ item.value }}
            </div>
            <div class="metric-label">{{ item.label }}</div>
          </el-card>
        </el-col>
      </el-row>

      <!-- 每类 AP 表格 -->
      <el-table
        :data="perClassData"
        stripe
        style="width: 100%; margin-top: 16px"
        :row-class-name="tableRowClassName"
      >
        <el-table-column prop="class_name" label="类别" width="200" />
        <el-table-column prop="ap50" label="AP@50" width="120">
          <template #default="{ row }">
            <span :style="{ color: row.ap50 < 0.5 ? '#f56c6c' : '#67c23a' }">
              {{ (row.ap50 * 100).toFixed(1) }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="ap50_95" label="AP@50-95" width="120">
          <template #default="{ row }">
            {{ (row.ap50_95 * 100).toFixed(1) }}%
          </template>
        </el-table-column>
        <el-table-column label="评价">
          <template #default="{ row }">
            <el-tag
              :type="row.ap50 >= 0.7 ? 'success' : row.ap50 >= 0.5 ? 'warning' : 'danger'"
              size="small"
            >
              {{ row.ap50 >= 0.7 ? '优秀' : row.ap50 >= 0.5 ? '良好' : '需改进' }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <!-- ── 导出模型对话框 ── -->
    <el-dialog v-model="showExportDialog" title="导出模型" width="500px">
      <el-form :model="exportForm" label-width="100px">
        <el-form-item label="版本号">
          <el-input v-model="exportForm.version" placeholder="自动生成（如 v1.0.0）" />
        </el-form-item>
        <el-form-item label="版本描述">
          <el-input
            v-model="exportForm.description"
            type="textarea"
            :rows="3"
            placeholder="描述本次训练的主要变更..."
          />
        </el-form-item>
        <el-form-item label="设为默认">
          <el-switch v-model="exportForm.set_default" />
          <span style="margin-left: 8px; color: #909399; font-size: 12px">
            设为该场景的默认检测模型
          </span>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showExportDialog = false">取消</el-button>
        <el-button type="primary" @click="exportModel" :loading="exporting">
          确认导出
        </el-button>
      </template>
    </el-dialog>

    <!-- ── 测试图验证对话框 ── -->
    <el-dialog v-model="showPredictDialog" title="测试图验证" width="900px">
      <el-row :gutter="16">
        <!-- 左侧：上传 + 配置 -->
        <el-col :span="10">
          <el-upload
            class="predict-upload"
            drag
            action=""
            :auto-upload="false"
            :on-change="handlePredictFileChange"
            accept="image/*"
            :limit="1"
          >
            <el-icon style="font-size: 40px; color: #909399"><UploadFilled /></el-icon>
            <div>拖拽图片到此处，或 <em>点击上传</em></div>
            <template #tip>
              <div class="el-upload__tip">支持 JPG/PNG/BMP 格式</div>
            </template>
          </el-upload>

          <el-form label-width="80px" style="margin-top: 16px">
            <el-form-item label="置信度">
              <el-slider v-model="predictConf" :min="0.05" :max="0.95" :step="0.05" show-input />
            </el-form-item>
            <el-form-item label="IoU">
              <el-slider v-model="predictIou" :min="0.1" :max="0.9" :step="0.05" show-input />
            </el-form-item>
          </el-form>

          <el-button
            type="primary"
            style="width: 100%; margin-top: 8px"
            @click="runPredict"
            :loading="predicting"
            :disabled="!predictFile"
          >
            开始检测
          </el-button>
        </el-col>

        <!-- 右侧：检测结果 -->
        <el-col :span="14">
          <div v-if="predictResult">
            <img
              :src="`data:image/jpeg;base64,${predictResult.annotated_image}`"
              style="width: 100%; border-radius: 8px; margin-bottom: 12px"
            />
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item label="检测目标数">
                {{ predictResult.total_objects }}
              </el-descriptions-item>
              <el-descriptions-item label="推理耗时">
                {{ predictResult.inference_time }}ms
              </el-descriptions-item>
            </el-descriptions>

            <el-table
              :data="predictResult.detections"
              stripe
              size="small"
              style="margin-top: 8px; max-height: 200px"
            >
              <el-table-column prop="class_name" label="类别" width="120" />
              <el-table-column label="置信度" width="100">
                <template #default="{ row }">
                  {{ (row.confidence * 100).toFixed(1) }}%
                </template>
              </el-table-column>
              <el-table-column label="位置">
                <template #default="{ row }">
                  [{{ row.bbox.map(v => v.toFixed(0)).join(', ') }}]
                </template>
              </el-table-column>
            </el-table>
          </div>
          <el-empty v-else description="上传图片并点击检测" />
        </el-col>
      </el-row>
    </el-dialog>

    <!-- 新建训练任务对话框 -->
    <el-dialog v-model="showCreateDialog" title="新建训练任务" width="600px" :close-on-click-modal="false">
      <el-form :model="trainForm" label-width="120px">
        <el-form-item label="检测场景">
          <el-select v-model="trainForm.scene_id" placeholder="选择场景">
            <el-option label="交通车辆检测" :value="1" />
            <el-option label="遥感目标检测" :value="2" />
            <el-option label="工业缺陷检测" :value="3" />
            <el-option label="农业病害检测" :value="4" />
          </el-select>
        </el-form-item>
        <el-form-item label="基础模型">
          <el-select v-model="trainForm.model_name">
            <el-option label="YOLOv11n (Nano, 最快)" value="yolov11n" />
            <el-option label="YOLOv11s (Small)" value="yolov11s" />
            <el-option label="YOLOv11m (Medium)" value="yolov11m" />
            <el-option label="YOLOv11l (Large)" value="yolov11l" />
            <el-option label="YOLOv11x (XLarge, 最强)" value="yolov11x" />
          </el-select>
        </el-form-item>
        <el-form-item label="训练轮数">
          <el-slider v-model="trainForm.epochs" :min="10" :max="500" :step="10" show-input />
        </el-form-item>
        <el-form-item label="批次大小">
          <el-input-number v-model="trainForm.batch_size" :min="1" :max="64" :step="2" />
        </el-form-item>
        <el-form-item label="图像尺寸">
          <el-select v-model="trainForm.img_size">
            <el-option label="416" :value="416" />
            <el-option label="512" :value="512" />
            <el-option label="640 (默认)" :value="640" />
            <el-option label="768" :value="768" />
          </el-select>
        </el-form-item>
        <el-form-item label="训练设备">
          <el-radio-group v-model="trainForm.device">
            <el-radio value="cpu">CPU (本地)</el-radio>
            <el-radio value="0">GPU:0</el-radio>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="优化器">
          <el-select v-model="trainForm.optimizer">
            <el-option label="SGD (推荐)" value="SGD" />
            <el-option label="Adam" value="Adam" />
            <el-option label="AdamW" value="AdamW" />
          </el-select>
        </el-form-item>
        <el-form-item label="初始学习率">
          <el-input-number v-model="trainForm.lr0" :min="0.0001" :max="0.1" :step="0.001" :precision="4" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="showCreateDialog = false">取消</el-button>
        <el-button type="primary" @click="createTask" :loading="creating">启动训练</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Refresh, UploadFilled } from '@element-plus/icons-vue'
import * as echarts from 'echarts'
import request from '@/utils/request'

const taskList = ref([])
const loadingTasks = ref(false)
const selectedTask = ref(null)
const showCreateDialog = ref(false)
const creating = ref(false)

const lossChartRef = ref(null)
const mapChartRef = ref(null)
let lossChart = null
let mapChart = null
let pollTimer = null

const trainForm = ref({
  scene_id: 1,
  model_name: 'yolov11n',
  epochs: 50,
  batch_size: 8,
  img_size: 640,
  device: 'cpu',
  optimizer: 'SGD',
  lr0: 0.01,
})

// ── 评估相关状态 ──
const evalReport = ref(null)
const validating = ref(false)

// ── 导出相关状态 ──
const showExportDialog = ref(false)
const exporting = ref(false)
const exportForm = ref({
  version: '',
  description: '',
  set_default: true,
})

// ── 测试验证相关状态 ──
const showPredictDialog = ref(false)
const predicting = ref(false)
const predictFile = ref(null)
const predictConf = ref(0.25)
const predictIou = ref(0.45)
const predictResult = ref(null)

// ── 评估报告指标卡片 ──
const evalMetricCards = computed(() => {
  if (!evalReport.value) return []
  const o = evalReport.value.overall
  return [
    {
      label: 'Precision',
      value: (o.precision * 100).toFixed(1) + '%',
      color: o.precision > 0.7 ? '#67c23a' : '#e6a23c',
    },
    {
      label: 'Recall',
      value: (o.recall * 100).toFixed(1) + '%',
      color: o.recall > 0.7 ? '#67c23a' : '#e6a23c',
    },
    {
      label: 'mAP@50',
      value: (o.map50 * 100).toFixed(1) + '%',
      color: o.map50 > 0.5 ? '#67c23a' : '#f56c6c',
    },
    {
      label: 'mAP@50-95',
      value: (o.map50_95 * 100).toFixed(1) + '%',
      color: o.map50_95 > 0.3 ? '#67c23a' : '#f56c6c',
    },
  ]
})

// ── 每类 AP 表格数据 ──
const perClassData = computed(() => {
  if (!evalReport.value || !evalReport.value.per_class) return []
  return Object.entries(evalReport.value.per_class)
    .map(([name, m]) => ({
      class_name: name,
      ap50: m.ap50,
      ap50_95: m.ap50_95,
    }))
    .sort((a, b) => b.ap50 - a.ap50)
})

// ── 表格行样式 ──
function tableRowClassName({ row }) {
  return row.ap50 < 0.5 ? 'weak-row' : ''
}

const metricCards = computed(() => {
  if (!selectedTask.value) return []
  const m = selectedTask.value.latest_metric
  if (!m) return [
    { label: 'Epoch', value: `${selectedTask.value.current_epoch || 0}/${selectedTask.value.epochs}` },
    { label: '进度', value: `${selectedTask.value.progress || 0}%` },
    { label: 'Box Loss', value: '-' },
    { label: 'Cls Loss', value: '-' },
    { label: 'mAP@50', value: '-' },
    { label: 'mAP@50-95', value: '-' },
  ]
  return [
    { label: 'Epoch', value: `${m.epoch}/${selectedTask.value.epochs}` },
    { label: 'Box Loss', value: m.box_loss != null ? m.box_loss.toFixed(4) : '-' },
    { label: 'Cls Loss', value: m.cls_loss != null ? m.cls_loss.toFixed(4) : '-' },
    { label: 'Precision', value: m.precision != null ? (m.precision * 100).toFixed(1) + '%' : '-' },
    { label: 'mAP@50', value: m.map50 != null ? (m.map50 * 100).toFixed(1) + '%' : '-' },
    { label: 'mAP@50-95', value: m.map50_95 != null ? (m.map50_95 * 100).toFixed(1) + '%' : '-' },
  ]
})

function statusType(status) {
  const map = { pending: 'info', running: 'warning', completed: 'success', failed: 'danger', cancelled: 'info' }
  return map[status] || 'info'
}

function statusText(status) {
  const map = { pending: '等待中', running: '训练中', completed: '已完成', failed: '失败', cancelled: '已取消' }
  return map[status] || status
}

async function fetchTasks() {
  loadingTasks.value = true
  try {
    const res = await request.get('/training/tasks')
    taskList.value = res.items || []
  } catch (e) {
    console.error('获取任务列表失败', e)
  } finally {
    loadingTasks.value = false
  }
}

async function selectTask(task) {
  selectedTask.value = task
  await nextTick()
  initCharts()
  fetchMetrics()
  startPolling()
}

function initCharts() {
  if (lossChart) lossChart.dispose()
  if (mapChart) mapChart.dispose()
  if (lossChartRef.value) lossChart = echarts.init(lossChartRef.value)
  if (mapChartRef.value) mapChart = echarts.init(mapChartRef.value)
}

async function fetchMetrics() {
  if (!selectedTask.value) return
  try {
    const taskId = selectedTask.value.id || selectedTask.value.task?.id
    const res = await request.get(`/training/metrics/${taskId}`)
    const metrics = res.metrics || []
    const statusRes = await request.get(`/training/status/${taskId}`)
    if (statusRes) {
      selectedTask.value = { ...selectedTask.value, ...statusRes }
    }
    if (metrics.length > 0) updateCharts(metrics)
  } catch (e) {
    console.error('获取训练指标失败', e)
  }
}

function updateCharts(metrics) {
  const epochs = metrics.map((m) => m.epoch)
  if (lossChart) {
    lossChart.setOption({
      title: { text: '训练损失曲线', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { data: ['Box Loss', 'Cls Loss', 'DFL Loss'], bottom: 0 },
      grid: { left: '10%', right: '5%', top: '15%', bottom: '15%' },
      xAxis: { type: 'category', data: epochs, name: 'Epoch' },
      yAxis: { type: 'value', name: 'Loss' },
      series: [
        { name: 'Box Loss', type: 'line', data: metrics.map((m) => m.box_loss), smooth: true, lineStyle: { width: 2 } },
        { name: 'Cls Loss', type: 'line', data: metrics.map((m) => m.cls_loss), smooth: true, lineStyle: { width: 2 } },
        { name: 'DFL Loss', type: 'line', data: metrics.map((m) => m.dfl_loss), smooth: true, lineStyle: { width: 2 } },
      ],
    })
  }
  if (mapChart) {
    mapChart.setOption({
      title: { text: '评估指标曲线', left: 'center', textStyle: { fontSize: 14 } },
      tooltip: { trigger: 'axis' },
      legend: { data: ['mAP@50', 'mAP@50-95', 'Precision', 'Recall'], bottom: 0 },
      grid: { left: '10%', right: '5%', top: '15%', bottom: '15%' },
      xAxis: { type: 'category', data: epochs, name: 'Epoch' },
      yAxis: { type: 'value', name: '指标值', max: 1 },
      series: [
        { name: 'mAP@50', type: 'line', data: metrics.map((m) => m.map50), smooth: true, lineStyle: { width: 2, color: '#409eff' }, itemStyle: { color: '#409eff' } },
        { name: 'mAP@50-95', type: 'line', data: metrics.map((m) => m.map50_95), smooth: true, lineStyle: { width: 2, color: '#67c23a' }, itemStyle: { color: '#67c23a' } },
        { name: 'Precision', type: 'line', data: metrics.map((m) => m.precision), smooth: true, lineStyle: { width: 2, type: 'dashed', color: '#e6a23c' }, itemStyle: { color: '#e6a23c' } },
        { name: 'Recall', type: 'line', data: metrics.map((m) => m.recall), smooth: true, lineStyle: { width: 2, type: 'dashed', color: '#f56c6c' }, itemStyle: { color: '#f56c6c' } },
      ],
    })
  }
}

function startPolling() {
  stopPolling()
  pollTimer = setInterval(() => { if (selectedTask.value) fetchMetrics() }, 5000)
}

function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

async function createTask() {
  creating.value = true
  try {
    const res = await request.post('/training/start', trainForm.value)
    ElMessage.success(`训练任务已创建：${res.task_uuid}`)
    showCreateDialog.value = false
    await fetchTasks()
    if (res.id) {
      const newTask = taskList.value.find((t) => t.id === res.id)
      if (newTask) selectTask(newTask)
    }
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '创建训练任务失败')
  } finally {
    creating.value = false
  }
}

async function stopTask(taskId) {
  try {
    await ElMessageBox.confirm('确定要停止当前训练任务吗？训练进度将被保留。', '确认停止', { type: 'warning' })
    await request.post(`/training/stop/${taskId}`)
    ElMessage.success('训练任务已停止')
    await fetchTasks()
  } catch (e) {
    if (e !== 'cancel') ElMessage.error('停止训练失败')
  }
}

// ── 评估模型 ──
async function validateModel() {
  if (!selectedTask.value) return
  validating.value = true
  try {
    const taskId = selectedTask.value.id || selectedTask.value.task?.id
    const res = await request.post(`/training/validate/${taskId}`, {
      split: 'val',
      conf: 0.001,
      iou: 0.6,
    })
    evalReport.value = res.data || res
    ElMessage.success(`评估完成: mAP50=${((evalReport.value.overall?.map50 || 0) * 100).toFixed(1)}%`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '模型评估失败')
  } finally {
    validating.value = false
  }
}

// ── 导出模型 ──
async function exportModel() {
  if (!selectedTask.value) return
  exporting.value = true
  try {
    const taskId = selectedTask.value.id || selectedTask.value.task?.id
    const res = await request.post(`/training/export/${taskId}`, exportForm.value)
    const data = res.data || res
    ElMessage.success(data.message || '模型导出成功')
    showExportDialog.value = false
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '模型导出失败')
  } finally {
    exporting.value = false
  }
}

// ── 下载模型 ──
async function downloadModel() {
  if (!selectedTask.value) return
  try {
    const taskId = selectedTask.value.id || selectedTask.value.task?.id
    const token = localStorage.getItem('token') || ''
    const response = await fetch(
      `/training/download/${taskId}`,
      { headers: { Authorization: `Bearer ${token}` } }
    )
    if (!response.ok) throw new Error('下载失败')
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `best_${selectedTask.value.task_uuid}.pt`
    document.body.appendChild(a)
    a.click()
    a.remove()
    window.URL.revokeObjectURL(url)
    ElMessage.success('模型下载已开始')
  } catch (e) {
    ElMessage.error('模型下载失败')
  }
}

// ── 测试图文件选择 ──
function handlePredictFileChange(file) {
  predictFile.value = file.raw
  predictResult.value = null
}

// ── 运行测试图预测 ──
async function runPredict() {
  if (!predictFile.value || !selectedTask.value) return
  predicting.value = true
  try {
    const taskId = selectedTask.value.id || selectedTask.value.task?.id
    const formData = new FormData()
    formData.append('file', predictFile.value)
    formData.append('task_id', taskId)
    formData.append('conf', predictConf.value)
    formData.append('iou', predictIou.value)

    const res = await request.post('/training/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    predictResult.value = res.data || res
    ElMessage.success(`检测完成: 发现 ${predictResult.value.total_objects} 个目标`)
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || '检测失败')
  } finally {
    predicting.value = false
  }
}

onMounted(() => { fetchTasks() })

onBeforeUnmount(() => {
  stopPolling()
  if (lossChart) lossChart.dispose()
  if (mapChart) mapChart.dispose()
})
</script>

<style scoped>
.training-page { padding: 20px; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
.page-header h2 { margin: 0; font-size: 22px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.monitor-info { display: flex; gap: 16px; font-size: 13px; color: #909399; }
.metric-cards { margin-bottom: 8px; }
.metric-item { text-align: center; padding: 8px 0; }
.metric-value { font-size: 20px; font-weight: 700; color: #303133; }
.metric-label { font-size: 12px; color: #909399; margin-top: 4px; }
.task-list-card, .monitor-card, .action-card, .eval-card { margin-bottom: 20px; }

.predict-upload { width: 100%; }
.predict-upload :deep(.el-upload-dragger) { width: 100%; padding: 20px; }

:deep(.weak-row) { background-color: #fef0f0 !important; }
:deep(.weak-row td) { color: #f56c6c !important; }
</style>
