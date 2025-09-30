import { readFileSync } from 'node:fs'
import { JSDOM } from 'jsdom'

function ok(cond, msg) {
  if (!cond) throw new Error(msg)
}

function main() {
  const html = readFileSync('web/index.html', 'utf-8')
  const dom = new JSDOM(html)
  const d = dom.window.document

  const ids = [
    // toolbar
    'btnLayout','btnAlignX','btnAlignY','btnSnap','gridSize','snapStrength','btnFit','btnClear','brandSelect','btnTheme','btnExport','btnDistX','btnDistY','btnTrainBt',
    // settings
    'nodeForm','apiUrl','applyApi','llmBaseUrl','llmModelSet','defaultTf','btnLoadSettings','btnSaveSettings','btnApplySettings',
    // common params
    'cfg','featureFile','timeframe','llmModel','llmCount','timerange','btnExpr','btnBacktest','btnSummary',
    // feature & charts
    'featTop','featChart',
    // flow
    'flowCfg','flowSteps','btnRunAgentFlow','flowStatus',
    // results
    'btnList','resA','resB','btnCompare','comparePanel','btnGallery','galleryPanel','aggNames','btnAgg','aggPanel','mvPanel',
    // canvas & logs
    'root','statusBar','btnCancelJob','logs','cards'
  ]
  const missing = ids.filter(id => !d.getElementById(id))
  if (missing.length) {
    console.error('[front-smoke] missing ids:', missing.join(', '))
    process.exit(1)
  }

  const tokens = [
    '节点面板','服务设置','常用参数','特征 TopN','结果','图集','聚合','多时段验证','日志'
  ]
  const text = d.body.textContent || ''
  const missTokens = tokens.filter(t => !text.includes(t))
  if (missTokens.length) {
    console.error('[front-smoke] missing labels:', missTokens.join(', '))
    process.exit(2)
  }

  console.log('[front-smoke] OK - all required ids & labels present')
}

main()

