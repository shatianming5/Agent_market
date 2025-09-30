// Extra enhancements loaded after app.js
// - Gallery virtualization rendering
// - ECharts instance复用与小修

(function(){
  function onReady(fn){ if(document.readyState==='complete'||document.readyState==='interactive'){ setTimeout(fn,0) } else { document.addEventListener('DOMContentLoaded', fn) } }

  // Patch alert to replace遗留乱码提示
  function patchAlert(){
    try {
      const old = window.alert
      window.alert = function(msg){
        try {
          if (typeof msg === 'string' && msg.indexOf('�') >= 0) {
            if (/2/.test(msg)) return old('请选择不少于2个节点')
            if (/3/.test(msg)) return old('请选择不少于3个节点')
            if (/health/i.test(msg)) return old('API 健康检查失败，请检查地址')
            return old('操作失败，请检查输入')
          }
        } catch {}
        return old(msg)
      }
    } catch {}
  }

  function getChartById(id){ try{ if(!window.echarts) return null; const el=document.getElementById(id); if(!el) return null; return (echarts.getInstanceByDom && echarts.getInstanceByDom(el)) || echarts.init(el) } catch(e){ return null } }

  function renderGalleryVirtual(container, items){
    const wrap = document.createElement('div'); wrap.style.maxHeight='420px'; wrap.style.overflowY='auto'; wrap.style.padding='6px'
    const inner = document.createElement('div'); inner.className='virt-list'; wrap.appendChild(inner)
    container.innerHTML=''; container.appendChild(wrap)
    const renderItem = (x) => {
      const pct = Number(x.profit_total_pct||0)
      const width = Math.max(0, Math.min(100, Math.round(Math.abs(pct))))
      const color = pct >= 0 ? 'linear-gradient(90deg, #86efac, #22c55e)' : 'linear-gradient(90deg, #fecaca, #ef4444)'
      return `<div class="mini-card"><div class="title">${x.name}</div><div class="row"><span>收益%</span><b>${x.profit_total_pct ?? '--'}</b></div><div class="row"><span>交易数</span><b>${x.trades ?? '--'}</b></div><div class="row"><span>最大回撤</span><b>${x.max_drawdown_abs ?? '--'}</b></div><div class="mini-bar"><i style="width:${width}%; background:${color}"></i></div></div>`
    }
    const sample = document.createElement('div'); sample.innerHTML = renderItem(items[0]||{}); sample.style.visibility='hidden'; document.body.appendChild(sample)
    const itemH = Math.max(72, sample.firstChild.offsetHeight || 90); try { document.body.removeChild(sample) } catch{}
    const viewportH = 420; const buffer = 6
    function render(){
      const scrollTop = wrap.scrollTop
      const visibleCount = Math.ceil(viewportH / itemH) + buffer
      const start = Math.max(0, Math.floor(scrollTop / itemH) - Math.floor(buffer/2))
      const end = Math.min(items.length, start + visibleCount)
      inner.innerHTML = items.slice(start, end).map(renderItem).join('') || '<em>暂无数据</em>'
    }
    wrap.addEventListener('scroll', render)
    render()
  }

  async function showSummaryPatched(){
    try {
      const r = await fetch(`${API}/results/latest-summary`)
      const data = await r.json()
      const summaryEl = document.getElementById('summary'); if (summaryEl) summaryEl.textContent = JSON.stringify(data, null, 2)
      if (Array.isArray(data.trades) && window.echarts) {
        const sorted = data.trades.slice().sort((a,b)=> (a.open_timestamp||0) - (b.open_timestamp||0))
        const xs = []; const ys = []; let cum = 0
        for (const t of sorted){ cum += Number(t.profit_abs||0); ys.push(cum); xs.push(new Date(t.open_timestamp||0).toISOString().slice(0,10)) }
        const el = document.getElementById('chart')
        if (el) {
          const chart = (window.echarts.getInstanceByDom && el) ? (echarts.getInstanceByDom(el) || echarts.init(el)) : echarts.init(el)
          chart.setOption({ grid:{left:40,right:16,top:10,bottom:30}, xAxis:{ type:'category', data:xs, axisLabel:{ rotate:45 } }, yAxis:{ type:'value', scale:true }, tooltip:{ trigger:'axis' }, series:[{ name:'累计收益(USDT)', type:'line', data:ys }] })
        }
      }
    } catch(e){ console.warn('showSummary patched error', e) }
  }

  function hook(){
    patchAlert()

    // Override applyApi to更友好提示
    const applyBtn = document.getElementById('applyApi')
    if (applyBtn) applyBtn.onclick = async () => {
      try {
        const val = (document.getElementById('apiUrl')?.value || '').trim(); if (!val) return
        setApiUrl(val)
        const r = await fetch(`${API}/health`); const j = await r.json()
        alert(`API 健康检查成功: ${API} /health: ${JSON.stringify(j)}`)
      } catch (e) { alert(`API 健康检查失败: ${API}，调用 /health 出错: ${e}`) }
    }

    // Override status renderer
    window.setStatus = function(phase, jobId, running){
      const el = document.getElementById('statusBar'); if (!el) return
      if (running) { el.className='status running'; el.innerHTML = `<i class="ri-loader-4-line spin"></i> ${phase} 运行中 · job ${jobId}` }
      else { el.className='status'; el.innerHTML = `<i class="ri-check-line"></i> ${phase} 已完成 · job ${jobId}` }
    }

    // Override showSummary for中文文案 + ECharts 实例复用
    window.showSummary = showSummaryPatched
    // Override Gallery to use virtualization and larger limit
    const gbtn = document.getElementById('btnGallery')
    if (gbtn) gbtn.onclick = async () => {
      try {
        const res = await fetch(`${API}/results/gallery?limit=500`)
        const data = await res.json()
        const gp = document.getElementById('galleryPanel')
        if (gp) renderGalleryVirtual(gp, data.items||[])
      } catch(e){ alert('加载图集失败: '+e) }
    }

    // Override Aggregate chart to reuse ECharts instance
    // Inject metric selector for聚合
    const aggNames = document.getElementById('aggNames')
    if (aggNames && !document.getElementById('aggMetric')){
      const sel = document.createElement('select'); sel.id='aggMetric'; sel.style.marginTop='6px'
      sel.innerHTML = '<option value="profit_total_pct">收益%</option><option value="trades">交易数</option><option value="max_drawdown_abs">最大回撤</option>'
      aggNames.parentElement && aggNames.parentElement.insertBefore(sel, aggNames.nextSibling)
    }
    const abtn = document.getElementById('btnAgg')
    if (abtn) abtn.onclick = async () => {
      try {
        const names = (document.getElementById('aggNames').value||'').trim(); if(!names) { alert('请先输入需聚合的结果名，如 a.zip,b.zip'); return }
        const res = await fetch(`${API}/results/aggregate?names=${encodeURIComponent(names)}`)
        const data = await res.json()
        const ap = document.getElementById('aggPanel')
        if (!ap) return
        ap.innerHTML = '<div id="aggChart" style="height:220px"></div><div id="aggInfo" class="panel" style="margin-top:6px"></div>'
        const chart = getChartById('aggChart')
        if (chart) {
          const items = data.items || []
          const metricSel = document.getElementById('aggMetric')
          const key = (metricSel && metricSel.value) || 'profit_total_pct'
          const y = items.map(i => (i[key] || 0))
          chart.setOption({ grid:{left:40,right:16,top:10,bottom:60}, xAxis:{ type:'category', data: items.map(i=>i.name), axisLabel:{ rotate:60 } }, yAxis:{ type:'value', scale:true }, tooltip:{ trigger:'axis' }, series:[{ type:'bar', data: y }] })
        }
        const info = document.getElementById('aggInfo'); if (info) info.innerHTML = `均值收益: ${data.mean_profit ?? '--'} | 收益波动: ${data.std_profit ?? '--'} | 稳健分: ${data.robust_score ?? '--'}`
      } catch(e){ alert('聚合失败: '+e) }
    }

    // Override compare to中文表头
    const cmp = document.getElementById('btnCompare')
    if (cmp) cmp.onclick = async () => {
      const a = (document.getElementById('resA').value||'').trim(); const b = (document.getElementById('resB').value||'').trim(); if (!a || !b) { alert('请先输入结果 A 与 B'); return }
      const [ra, rb] = await Promise.all([
        fetch(`${API}/results/summary?name=${encodeURIComponent(a)}`),
        fetch(`${API}/results/summary?name=${encodeURIComponent(b)}`),
      ])
      const [sa, sb] = [await ra.json(), await rb.json()]
      const toM = (s) => { const c = Array.isArray(s.strategy_comparison) && s.strategy_comparison.length ? s.strategy_comparison[0] : {}; return { profit_pct: c.profit_total_pct ?? s.profit_total_pct, profit_abs: c.profit_total_abs ?? s.profit_total_abs, trades: c.trades ?? s.trades, winrate: c.winrate ?? s.winrate, max_dd: c.max_drawdown_abs ?? s.max_drawdown_abs } }
      const ma = toM(sa), mb = toM(sb)
      const el = document.getElementById('comparePanel')
      if (el) el.innerHTML = `
        <table style="width:100%; font-size:12px; border-collapse:collapse">
          <tr><th>指标</th><th>A (${a})</th><th>B (${b})</th></tr>
          <tr><td>收益%</td><td>${ma.profit_pct ?? '--'}</td><td>${mb.profit_pct ?? '--'}</td></tr>
          <tr><td>收益(USDT)</td><td>${ma.profit_abs ?? '--'}</td><td>${mb.profit_abs ?? '--'}</td></tr>
          <tr><td>交易数</td><td>${ma.trades ?? '--'}</td><td>${mb.trades ?? '--'}</td></tr>
          <tr><td>胜率</td><td>${ma.winrate ?? '--'}</td><td>${mb.winrate ?? '--'}</td></tr>
          <tr><td>最大回撤(USDT)</td><td>${ma.max_dd ?? '--'}</td><td>${mb.max_dd ?? '--'}</td></tr>
        </table>`
    }
  }

  onReady(() => setTimeout(hook, 50))
})()
