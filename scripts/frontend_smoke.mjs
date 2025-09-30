import { readFileSync } from 'node:fs'
import { JSDOM } from 'jsdom'

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

  const text = (d.body && d.body.textContent) || ''
  // Fail if we detect any Unicode replacement characters (mojibake)
  if (/\uFFFD/.test(text)) {
    console.error('[front-smoke] mojibake detected in page text')
    process.exit(2)
  }

  console.log('[front-smoke] OK - structure present, no mojibake detected')
}

main()

