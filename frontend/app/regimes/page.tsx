'use client'

import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function RegimesPage() {
  const [regimes, setRegimes] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [selectedAsset, setSelectedAsset] = useState('')

  useEffect(() => {
    fetchRegimes()
  }, [])

  const fetchRegimes = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/regimes/`)
      if (res.ok) {
        const data = await res.json()
        setRegimes(data)
        if (data.asset_ids && data.asset_ids.length > 0) {
          setSelectedAsset(data.asset_ids[0])
        }
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Format chart data for selected asset
  const getRegimeChartData = () => {
    if (!regimes || !selectedAsset) return []
    const assetIdx = regimes.asset_ids.indexOf(selectedAsset)
    if (assetIdx === -1) return []
    
    const dates = regimes.dates
    const probs = regimes.regime_probs[assetIdx]  // (T, 4)
    
    // For visualization density, sample last 30 periods
    const lastN = 30
    const startIdx = Math.max(0, dates.length - lastN)
    
    return dates.slice(startIdx).map((date: string, idx: number) => {
      const actualIdx = startIdx + idx
      const probVec = probs[actualIdx]
      return {
        date: date.split(' ')[0],
        R1: probVec[0],
        R2: probVec[1],
        R3: probVec[2],
        R4: probVec[3],
      }
    })
  }

  const chartData = getRegimeChartData()

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Regime Classifier</h1>
        <p className="text-slate-400">Continuous classification of assets along memory and credit latency axes</p>
      </div>

      {loading ? (
        <div className="text-slate-500 py-12 text-center">Querying model regimes...</div>
      ) : regimes ? (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar selector */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-1 max-h-screen overflow-y-auto">
            <h2 className="text-xl font-semibold mb-4 text-blue-400">Asset Selection</h2>
            <div className="space-y-2">
              {regimes.asset_ids.map((asset: string) => (
                <button
                  key={asset}
                  onClick={() => setSelectedAsset(asset)}
                  className={`w-full text-left px-4 py-2 rounded transition font-medium ${
                    selectedAsset === asset ? 'bg-blue-600 text-white' : 'hover:bg-slate-700 text-slate-300'
                  }`}
                >
                  {asset}
                </button>
              ))}
            </div>
          </div>

          {/* Visualizer */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-3">
            <div className="mb-6 flex justify-between items-center">
              <h2 className="text-xl font-semibold text-blue-400">Regime Allocation history ({selectedAsset})</h2>
              <div className="flex gap-2">
                <span className="text-xs font-semibold px-2 py-1 bg-blue-900/30 text-blue-300 rounded border border-blue-800">
                  R1: Long/Long
                </span>
                <span className="text-xs font-semibold px-2 py-1 bg-cyan-900/30 text-cyan-300 rounded border border-cyan-800">
                  R2: Long/Short
                </span>
                <span className="text-xs font-semibold px-2 py-1 bg-amber-900/30 text-amber-300 rounded border border-amber-800">
                  R3: Short/Long
                </span>
                <span className="text-xs font-semibold px-2 py-1 bg-red-900/30 text-red-300 rounded border border-red-800">
                  R4: Short/Short
                </span>
              </div>
            </div>

            {chartData.length > 0 ? (
              <div className="h-96">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid stroke="#374151" />
                    <XAxis stroke="#6b7280" dataKey="date" />
                    <YAxis stroke="#6b7280" domain={[0, 1]} />
                    <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                    <Legend />
                    <Bar dataKey="R1" stackId="a" fill="#3b82f6" name="R1: Trending + Slow Macro" />
                    <Bar dataKey="R2" stackId="a" fill="#06b6d4" name="R2: Trending + Instant Reaction" />
                    <Bar dataKey="R3" stackId="a" fill="#f59e0b" name="R3: Mean Reverting + Slow Macro" />
                    <Bar dataKey="R4" stackId="a" fill="#ef4444" name="R4: Mean Reverting + Instant" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-96 flex items-center justify-center text-slate-500">
                Chart details unavailable.
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="text-slate-500 py-12 text-center">No regime data available. Upload data first.</div>
      )}
    </div>
  )
}
