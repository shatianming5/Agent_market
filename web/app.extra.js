// Extra enhancements loaded after app.js
// - Gallery virtualization rendering
// - ECharts instance复用与小修

(function(){
  function onReady(fn){ if(document.readyState==='complete'||document.readyState==='interactive'){ setTimeout(fn,0) } else { document.addEventListener('DOMContentLoaded', fn) } }

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

  function hook(){
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
          chart.setOption({ grid:{left:40,right:16,top:10,bottom:60}, xAxis:{ type:'category', data: items.map(i=>i.name), axisLabel:{ rotate:60 } }, yAxis:{ type:'value', scale:true }, tooltip:{ trigger:'axis' }, series:[{ type:'bar', data: items.map(i=>i.profit_total_pct||0) }] })
        }
        const info = document.getElementById('aggInfo'); if (info) info.innerHTML = `均值收益: ${data.mean_profit ?? '--'} | 收益波动: ${data.std_profit ?? '--'} | 稳健分: ${data.robust_score ?? '--'}`
      } catch(e){ alert('聚合失败: '+e) }
    }

    // Minor: better prompt for compare
    const cmp = document.getElementById('btnCompare')
    if (cmp) {
      const old = cmp.onclick
      cmp.onclick = async () => {
        const a = (document.getElementById('resA').value||'').trim(); const b = (document.getElementById('resB').value||'').trim(); if (!a || !b) { alert('请先输入结果 A 与 B'); return }
        if (typeof old === 'function') return old()
      }
    }
  }

  onReady(() => setTimeout(hook, 50))
})()

