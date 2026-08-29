'use client'

import { useState, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function PredictionsPage() {
  const [predictions, setPredictions] = useState<any>(null)
  const [targetRtg, setTargetRtg] = useState(0.05)
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')

  useEffect(() => {
    fetchPredictions()
  }, [])

  const fetchPredictions = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/predict/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_rtg: targetRtg })
      })
      if (res.ok) {
        const data = await res.json()
        setPredictions(data)
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Format data for display: we take the last timestep predictions
  const getAssetRows = () => {
    if (!predictions) return []
    const rows = []
    const assetIds = predictions.asset_ids
    const T = predictions.timestamps.length
    
    for (let i = 0; i < assetIds.length; i++) {
      const asset = assetIds[i]
      const regimeIndex = predictions.regimes[i][T - 1]
      const regimeName = `R${regimeIndex + 1}`
      
      let val = 0.0
      let confidence = 0.0
      let typeLabel = ''
      
      if (predictions.portfolio_weights) {
        val = predictions.portfolio_weights[i][T - 1][i] // Allocation of asset i in asset i's model
        confidence = predictions.confidence[i][T - 1]
        typeLabel = 'Weight'
      } else if (predictions.forecasts) {
        // forecasts is (N, T, N, horizon), take self-prediction at last timestep
        val = predictions.forecasts[i][T - 1][i][0] // first step forecast of asset i in asset i's model
        confidence = predictions.confidence[i][T - 1][i]
        typeLabel = 'Forecast'
      }
      
      rows.push({
        asset,
        value: val,
        confidence,
        regime: regimeName,
        typeLabel,
      })
    }
    
    return rows.filter(r => r.asset.toLowerCase().includes(searchTerm.toLowerCase()))
  }

  const rows = getAssetRows()

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Predictions</h1>
        <p className="text-slate-400">Model outputs, asset weights, forecasts, and confidence metrics</p>
      </div>

      {/* Inputs */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 mb-8 flex flex-col md:flex-row gap-6 items-end">
        <div>
          <label className="block text-xs uppercase text-slate-400 mb-1">Target Return-To-Go (Conditioning)</label>
          <input 
            type="number" 
            step="0.01" 
            value={targetRtg} 
            onChange={(e) => setTargetRtg(Number(e.target.value))}
            className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white w-48"
          />
        </div>
        <div className="flex-1">
          <label className="block text-xs uppercase text-slate-400 mb-1">Search Asset</label>
          <input 
            type="text" 
            placeholder="Search symbol (e.g. AAPL)..."
            value={searchTerm} 
            onChange={(e) => setSearchTerm(e.target.value)}
            className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white w-full"
          />
        </div>
        <button 
          onClick={fetchPredictions} 
          disabled={loading}
          className="btn-primary w-full md:w-auto disabled:opacity-50"
        >
          {loading ? 'Running...' : 'Update Predictions'}
        </button>
      </div>

      {/* Predictions Table */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4 text-blue-400">Latest Predictions (Last Timestep)</h2>
        
        {loading ? (
          <div className="text-slate-500 py-8 text-center">Loading model predictions...</div>
        ) : rows.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-300">
              <thead>
                <tr className="border-b border-slate-700 text-slate-400 font-semibold">
                  <th className="py-3">Asset Symbol</th>
                  <th className="py-3">{rows[0]?.typeLabel || 'Value'}</th>
                  <th className="py-3">Model Confidence</th>
                  <th className="py-3">Market Regime</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row: any) => (
                  <tr key={row.asset} className="border-b border-slate-700 hover:bg-slate-700/50">
                    <td className="py-3 font-semibold">{row.asset}</td>
                    <td className={`py-3 font-semibold ${row.value > 0 ? 'text-green-400' : row.value < 0 ? 'text-red-400' : 'text-slate-300'}`}>
                      {row.typeLabel === 'Weight' ? `${(row.value * 100).toFixed(2)}%` : row.value.toFixed(4)}
                    </td>
                    <td className="py-3">{(row.confidence * 100).toFixed(1)}%</td>
                    <td className="py-3">
                      <span className={`px-2 py-0.5 rounded text-xs font-semibold ${
                        row.regime === 'R1' ? 'bg-blue-900 text-blue-300' :
                        row.regime === 'R2' ? 'bg-cyan-900 text-cyan-300' :
                        row.regime === 'R3' ? 'bg-amber-900 text-amber-300' :
                        'bg-red-900 text-red-300'
                      }`}>
                        {row.regime}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-slate-500 py-8 text-center">No predictions available. Ensure you have loaded data and trained the model.</div>
        )}
      </div>
    </div>
  )
}
