'use client'

import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function BacktestPage() {
  const [backtest, setBacktest] = useState<any>(null)
  const [txCosts, setTxCosts] = useState(10.0)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    runBacktest()
  }, [])

  const runBacktest = async () => {
    setLoading(true)
    setMessage('')
    try {
      const res = await fetch(`${API_URL}/backtest/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transaction_cost_bps: txCosts })
      })
      if (res.ok) {
        const data = await res.json()
        setBacktest(data)
      } else {
        const err = await res.json()
        setMessage(`Error: ${err.detail}`)
      }
    } catch (err: any) {
      setMessage(`Failed: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Format data for chart
  const getChartData = () => {
    if (!backtest) return []
    return backtest.dates.map((date: string, idx: number) => ({
      date: date.split(' ')[0],
      portfolio: backtest.equity_curve[idx],
      benchmark: backtest.benchmark_curve[idx],
    }))
  }

  const chartData = getChartData()

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Backtest Engine</h1>
        <p className="text-slate-400">Evaluate historical strategy performance and risk statistics</p>
      </div>

      {/* Configuration */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 mb-8 flex flex-col md:flex-row gap-6 items-end">
        <div>
          <label className="block text-xs uppercase text-slate-400 mb-1">Transaction Costs (BPS per trade)</label>
          <input 
            type="number" 
            value={txCosts} 
            onChange={(e) => setTxCosts(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white w-48"
          />
        </div>
        <button 
          onClick={runBacktest} 
          disabled={loading}
          className="btn-primary w-full md:w-auto disabled:opacity-50"
        >
          {loading ? 'Simulating...' : 'Run Simulation'}
        </button>
      </div>

      {message && (
        <div className="mb-6 p-4 rounded bg-red-900/30 text-red-400 text-sm">
          {message}
        </div>
      )}

      {/* Metrics */}
      {backtest && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-8">
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Cumulative Return</div>
            <div className={`metric-value ${backtest.cumulative_return > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {(backtest.cumulative_return * 100).toFixed(2)}%
            </div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Sharpe Ratio</div>
            <div className="metric-value">{backtest.sharpe_ratio.toFixed(2)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Sortino Ratio</div>
            <div className="metric-value">{backtest.sortino_ratio.toFixed(2)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Max Drawdown</div>
            <div className="metric-value text-red-400">{(backtest.max_drawdown * 100).toFixed(2)}%</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Volatility (Ann.)</div>
            <div className="metric-value">{(backtest.volatility * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Annualized Return</div>
            <div className="metric-value font-semibold">{(backtest.annualized_return * 100).toFixed(2)}%</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Strategy Turnover</div>
            <div className="metric-value">{backtest.turnover.toFixed(2)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Est. Tx Costs</div>
            <div className="metric-value font-semibold text-slate-400">{(backtest.transaction_costs * 100).toFixed(3)}%</div>
          </div>
        </div>
      )}

      {/* Chart */}
      {backtest && chartData.length > 0 ? (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 h-96">
          <h3 className="text-lg font-semibold mb-4">Historical Equity Curve</h3>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid stroke="#374151" />
              <XAxis stroke="#6b7280" dataKey="date" />
              <YAxis stroke="#6b7280" />
              <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
              <Legend />
              <Line type="monotone" dataKey="portfolio" stroke="#3b82f6" name="DependencyAI Portfolio" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="benchmark" stroke="#6b7280" name="Equal-Weighted Benchmark" dot={false} strokeDasharray="5 5" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="h-96 flex items-center justify-center border border-slate-700 rounded bg-slate-900/30 text-slate-500">
          Run backtest simulation to visualize performance.
        </div>
      )}
    </div>
  )
}
