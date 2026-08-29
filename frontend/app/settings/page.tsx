'use client'

import { useState } from 'react'

export default function SettingsPage() {
  const [leverageLimit, setLeverageLimit] = useState(2.0)
  const [defaultTxCost, setDefaultTxCost] = useState(10.0)
  const [maxWeightLimit, setMaxWeightLimit] = useState(0.1)
  const [serverUrl, setServerUrl] = useState('http://localhost:8000')

  const handleSave = () => {
    alert('Settings saved locally.')
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">System Settings</h1>
        <p className="text-slate-400">Configure default portfolio constraints and backend API endpoints</p>
      </div>

      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 max-w-2xl">
        <h2 className="text-xl font-semibold mb-6 text-blue-400">Portfolio Execution Settings</h2>
        
        <div className="space-y-6">
          <div>
            <label className="block text-xs uppercase text-slate-400 mb-1">Max Weight per Asset</label>
            <input 
              type="number" 
              step="0.01" 
              value={maxWeightLimit} 
              onChange={(e) => setMaxWeightLimit(Number(e.target.value))}
              className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white w-full"
            />
            <span className="text-xs text-slate-500 mt-1 block">Maximum allocation limit per asset in the portfolio</span>
          </div>

          <div>
            <label className="block text-xs uppercase text-slate-400 mb-1">Total Leverage Limit</label>
            <input 
              type="number" 
              step="0.1" 
              value={leverageLimit} 
              onChange={(e) => setLeverageLimit(Number(e.target.value))}
              className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white w-full"
            />
            <span className="text-xs text-slate-500 mt-1 block">Sum of absolute portfolio weights limit (L1 Norm constraint)</span>
          </div>

          <div>
            <label className="block text-xs uppercase text-slate-400 mb-1">Default Transaction Cost (BPS)</label>
            <input 
              type="number" 
              value={defaultTxCost} 
              onChange={(e) => setDefaultTxCost(Number(e.target.value))}
              className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white w-full"
            />
            <span className="text-xs text-slate-500 mt-1 block">Default fee per trade transaction in basis points (1 BPS = 0.01%)</span>
          </div>

          <div>
            <label className="block text-xs uppercase text-slate-400 mb-1">FastAPI Server URL</label>
            <input 
              type="text" 
              value={serverUrl} 
              onChange={(e) => setServerUrl(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white w-full"
            />
          </div>

          <button onClick={handleSave} className="btn-primary w-full md:w-auto">
            Save Settings
          </button>
        </div>
      </div>
    </div>
  )
}
