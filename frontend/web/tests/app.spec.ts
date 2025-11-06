import { test, expect } from '@playwright/test'
import type { Page } from '@playwright/test'

const BASE_URL = process.env.WEB_BASE_URL || 'http://127.0.0.1:5174'
const API_BASE_URL = process.env.API_BASE_URL || 'http://127.0.0.1:8032'

const EXPRESSIONS_FIXTURE = [
  { name: 'alpha_diff', expression: 'ema(close, 12) - ema(close, 26)', enabled: true },
  { name: 'beta_vol', expression: 'zscore(volume, 20)', enabled: false },
]

const BACKTEST_SUMMARY_FIXTURE = {
  source: 'backtest-result-mock.zip',
  strategy: 'ExpressionLongStrategy',
  metrics: {
    stake_currency: 'USDT',
    starting_balance: 1000,
    dry_run_wallet: 1000,
    final_balance: 1080,
    profit_total_pct: 8.0,
    profit_total_abs: 80.0,
    profit_mean: 0.011,
    profit_factor: 1.82,
    winrate: 0.6,
    expectancy: 0.12,
    sharpe: 1.4,
    trades_per_day: 2.1,
    max_drawdown_abs: 120.5,
    max_drawdown_account: 0.12,
    max_consecutive_wins: 4,
    max_consecutive_losses: 2,
    trades: [
      {
        pair: 'BTC/USDT',
        profit_ratio: 0.05,
        profit_abs: 50,
        trade_duration: 360,
        open_timestamp: 1700000000000,
        close_timestamp: 1700003600000,
        exit_reason: 'roi',
      },
      {
        pair: 'ETH/USDT',
        profit_ratio: -0.02,
        profit_abs: -20,
        trade_duration: 480,
        open_timestamp: 1700003600000,
        close_timestamp: 1700008400000,
        exit_reason: 'stop_loss',
      },
      {
        pair: 'ADA/USDT',
        profit_ratio: 0.03,
        profit_abs: 30,
        trade_duration: 240,
        open_timestamp: 1700008400000,
        close_timestamp: 1700010800000,
        exit_reason: 'exit_signal',
      },
    ],
    exit_reason_summary: [
      { key: 'roi', trades: 1, profit_total_pct: 5, winrate: 0.9 },
      { key: 'stop_loss', trades: 1, profit_total_pct: -2, winrate: 0.0 },
      { key: 'exit_signal', trades: 1, profit_total_pct: 3, winrate: 0.5 },
    ],
    daily_profit: [
      ['2021-01-01', 1.5],
      ['2021-01-02', -0.5],
      ['2021-01-03', 2.0],
    ],
  },
  pairs: [
    { pair: 'BTC/USDT', trades: 2, profit_total_pct: 6.0 },
    { pair: 'ETH/USDT', trades: 1, profit_total_pct: -2.0 },
    { pair: 'ADA/USDT', trades: 1, profit_total_pct: 4.0 },
  ],
  comparison: [
    { key: 'ExpressionLongStrategy', trades: 4, profit_total_pct: 8.0, winrate: 0.6 },
    { key: 'Baseline', trades: 4, profit_total_pct: 2.5, winrate: 0.45 },
  ],
}

async function stubJobs(page: Page) {
  await page.route('**/jobs', async (route) => {
    const url = new URL(route.request().url())
    if (url.pathname === '/jobs' && route.request().method() === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '[]',
      })
    } else {
      await route.continue()
    }
  })
}

async function stubBacktestFlow(page: Page, summary: unknown) {
  await page.route('**/run/backtest', async (route) => {
    if (route.request().method() === 'POST') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok', job_id: 'BT-123' }),
      })
    } else {
      await route.continue()
    }
  })

  await page.route('**/jobs/BT-123/progress', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ percent: 100, label: 'Complete', running: false, elapsed: 4 }),
    })
  })

  await page.route('**/jobs/BT-123/steps', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ steps: [{ idx: 1, total: 1, label: 'backtest', duration: 12 }] }),
    })
  })

  await page.route('**/steps/stats', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ stats: [] }),
    })
  })

  await page.route('**/backtest/summary/latest', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(summary),
    })
  })
}

async function stubExpressions(page: Page, expressions: unknown[]) {
  await page.route('**/expressions/allowed', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ functions: ['ema', 'sma'], base_columns: ['close', 'volume'] }),
    })
  })

  await page.route('**/expressions', async (route) => {
    const url = route.request().url()
    if (url.includes('/expressions/allowed')) {
      await route.continue()
      return
    }
    const method = route.request().method()
    if (method === 'GET') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ path: 'user_data/freqai_expressions.json', expressions }),
      })
    } else if (method === 'PUT') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ status: 'ok' }),
      })
    } else {
      await route.continue()
    }
  })
}

test.describe('Agent Market UI', () => {
  test.beforeEach(async ({ page }) => {
    await stubJobs(page)
  })

  test('loads dashboard tabs and hits health endpoint', async ({ page }) => {
    await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' })
    await expect(page.getByRole('button', { name: /Data Lab/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Expression Studio/ })).toBeVisible()
    await expect(page.getByRole('button', { name: /Job Monitor/ })).toBeVisible()

    const response = await page.request.get(`${API_BASE_URL}/health`)
    expect(response.status()).toBe(200)
  })

  test('renders backtest analytics summary after a mocked run', async ({ page }) => {
    await stubBacktestFlow(page, BACKTEST_SUMMARY_FIXTURE)
    await page.goto(`${BASE_URL}/`, { waitUntil: 'networkidle' })
    await page.locator('button:has-text("Backtests & Flows")').click()
    await page.locator('button:has-text("Run Backtest")').click()

    await expect(page.locator('text=Latest Backtest')).toBeVisible()
    await expect(page.locator('text=Equity Curve')).toBeVisible()
    await expect(page.locator('text=Exit Contribution')).toBeVisible()
    await expect(page.locator('text=Pair Performance')).toBeVisible()
    await expect(page.locator('text=Top Trades')).toBeVisible()
    await expect(page.locator('text=Risk Metrics')).toBeVisible()
  })

})
