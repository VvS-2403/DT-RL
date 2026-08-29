'use client'

import { useEffect, useState } from 'react'
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function Dashboard() {
  const [systemHealth, setSystemHealth] = useState<any>(null)
  const [equityCurveData, setEquityCurveData] = useState<any[]>([])
  const [regimeData, setRegimeData] = useState<any>(null)
  const [metrics, setMetrics] = useState<any>(null)
  const [predictions, setPredictions] = useState<any[]>([])

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      // Fetch system health
      const healthRes = await fetch(`${API_URL}/health`)
      if (healthRes.ok) {
        const health = await healthRes.json()
        setSystemHealth(health)
      } else {
        setSystemHealth({
          status: 'error',
          error_message: `HTTP ${healthRes.status}: ${healthRes.statusText}`
        })
      }

      // Mock equity curve data
      const equityData = Array.from({ length: 30 }, (_, i) => ({
        day: i + 1,
        value: 100000 + Math.random() * 50000 - 25000,
        benchmark: 100000 + i * 500,
      }))
      setEquityCurveData(equityData)

      // Mock regime distribution
      setRegimeData([
        { name: 'R1: Long/Long', value: 35, color: '#3b82f6' },
        { name: 'R2: Long/Short', value: 25, color: '#06b6d4' },
        { name: 'R3: Short/Long', value: 20, color: '#f59e0b' },
        { name: 'R4: Short/Short', value: 20, color: '#ef4444' },
      ])

      // Mock metrics
      setMetrics({
        cumulativeReturn: 0.242,
        sharpeRatio: 1.85,
        sortino: 2.41,
        maxDrawdown: -0.089,
        volatility: 0.156,
        numAssets: 50,
      })

      // Mock latest predictions
      const mockPredictions = ['ASSET_001', 'ASSET_002', 'ASSET_003', 'ASSET_004', 'ASSET_005'].map((asset) => ({
        asset,
        predictedReturn: Math.random() * 0.1 - 0.05,
        confidence: Math.random() * 0.3 + 0.7,
        regime: `R${Math.floor(Math.random() * 4) + 1}`,
      }))
      setPredictions(mockPredictions)
    } catch (err: any) {
      console.error('Failed to fetch dashboard data:', err)
      setSystemHealth({
        status: 'error',
        error_message: 'Failed to connect to backend server. Make sure the FastAPI server is running.'
      })
    }
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Dashboard</h1>
        <p className="text-slate-400">Real-time system overview and key metrics</p>
      </div>

      {/* Connection Error Banner */}
      {systemHealth && systemHealth.status === 'error' && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg text-red-200 text-sm">
          <span className="font-bold">⚠ Backend Connection Error:</span> {systemHealth.error_message}
          <div className="mt-2 text-xs text-slate-400">
            Current Endpoint Configured: <code className="bg-slate-900 px-1 py-0.5 rounded">{API_URL}</code>. 
            Make sure your FastAPI server is running on this address.
          </div>
        </div>
      )}

      {/* System Status */}
      {systemHealth && systemHealth.status === 'healthy' && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">System Status</div>
            <div className="metric-value">{systemHealth.status === 'healthy' ? '✓ Healthy' : '✗ Error'}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">GPU Available</div>
            <div className="metric-value">{systemHealth.gpu_available ? 'Yes' : 'CPU Only'}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Pipeline Loaded</div>
            <div className="metric-value">{systemHealth.pipeline_loaded ? 'Yes' : 'No'}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">PyTorch</div>
            <div className="metric-value text-base">v{systemHealth.pytorch_version?.split('.').slice(0, 2).join('.')}</div>
          </div>
        </div>
      )}

      {/* Performance Metrics */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Cumulative Return</div>
            <div className={`metric-value ${metrics.cumulativeReturn > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {(metrics.cumulativeReturn * 100).toFixed(2)}%
            </div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Sharpe Ratio</div>
            <div className="metric-value">{metrics.sharpeRatio.toFixed(2)}</div>
          </div>
          <div className="metric-card">
            <div className="text-xs text-slate-500 uppercase tracking-wide">Max Drawdown</div>
            <div className="metric-value text-red-400">{(metrics.maxDrawdown * 100).toFixed(2)}%</div>
          </div>
        </div>
      )}

      {/* Equity Curve */}
      {equityCurveData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="chart-container">
            <h3 className="text-lg font-semibold mb-4">Equity Curve</h3>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={equityCurveData}>
                <CartesianGrid stroke="#374151" />
                <XAxis stroke="#6b7280" dataKey="day" />
                <YAxis stroke="#6b7280" />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                <Legend />
                <Line type="monotone" dataKey="value" stroke="#3b82f6" dot={false} />
                <Line type="monotone" dataKey="benchmark" stroke="#6b7280" dot={false} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Regime Distribution */}
          {regimeData && (
            <div className="chart-container">
              <h3 className="text-lg font-semibold mb-4">Regime Distribution (Current)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={regimeData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name} ${value}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {regimeData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Latest Predictions */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Latest Predictions</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-2 text-slate-400">Asset</th>
              <th className="text-left py-2 text-slate-400">Predicted Return</th>
              <th className="text-left py-2 text-slate-400">Confidence</th>
              <th className="text-left py-2 text-slate-400">Regime</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((pred) => (
              <tr key={pred.asset} className="border-b border-slate-700 hover:bg-slate-700/50">
                <td className="py-2">{pred.asset}</td>
                <td className={`py-2 ${pred.predictedReturn > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {pred.predictedReturn.toFixed(3)}
                </td>
                <td className="py-2">{pred.confidence.toFixed(2)}</td>
                <td className="py-2">{pred.regime}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Quick Actions */}
      <div className="mt-8 flex gap-4">
        <button className="btn-primary">Load Data</button>
        <button className="btn-primary">Start Training</button>
        <button className="btn-secondary">Run Backtest</button>
        <button className="btn-secondary">Export Report</button>
      </div>
    </div>
  )
}
