'use client'

import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function TrainPage() {
  const [epochs, setEpochs] = useState(10)
  const [lr, setLr] = useState(0.001)
  const [batchSize, setBatchSize] = useState(16)
  const [outputMode, setOutputMode] = useState('portfolio_weights')
  const [progress, setProgress] = useState<any>(null)
  const [isTraining, setIsTraining] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    // Check initial training status
    fetchProgress()
    const timer = setInterval(fetchProgress, 2000)
    return () => clearInterval(timer)
  }, [])

  const fetchProgress = async () => {
    try {
      const res = await fetch(`${API_URL}/train/progress`)
      if (res.ok) {
        const data = await res.json()
        setProgress(data)
        setIsTraining(data.status === 'training')
      }
    } catch (err) {
      console.error(err)
    }
  }

  const handleStartTraining = async () => {
    setMessage('')
    try {
      const res = await fetch(`${API_URL}/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_epochs: epochs,
          learning_rate: lr,
          batch_size: batchSize,
          output_mode: outputMode,
        })
      })
      const result = await res.json()
      if (res.ok) {
        setMessage('Training started successfully!')
        setIsTraining(true)
      } else {
        setMessage(`Error: ${result.detail}`)
      }
    } catch (err: any) {
      setMessage(`Failed: ${err.message}`)
    }
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Model Training</h1>
        <p className="text-slate-400">Configure parameters and launch deep learning training pipeline</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configurations */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-1">
          <h2 className="text-xl font-semibold mb-4 text-blue-400">Hyperparameters</h2>
          
          <div className="space-y-4 mb-6">
            <div>
              <label className="block text-xs uppercase text-slate-400 mb-1">Epochs</label>
              <input 
                type="number" 
                value={epochs} 
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white" 
                min="1" 
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-xs uppercase text-slate-400 mb-1">Learning Rate</label>
              <input 
                type="number" 
                step="0.0001" 
                value={lr} 
                onChange={(e) => setLr(Number(e.target.value))}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white"
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-xs uppercase text-slate-400 mb-1">Batch Size</label>
              <input 
                type="number" 
                value={batchSize} 
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white"
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-xs uppercase text-slate-400 mb-1">Output Execution Mode</label>
              <select 
                value={outputMode} 
                onChange={(e) => setOutputMode(e.target.value)}
                className="w-full bg-slate-700 border border-slate-600 rounded px-3 py-2 text-white"
                disabled={isTraining}
              >
                <option value="portfolio_weights">Portfolio Weight Allocations (Direct)</option>
                <option value="forecasting">Stock Return Forecasting (Separated)</option>
              </select>
            </div>
          </div>

          <button 
            onClick={handleStartTraining} 
            disabled={isTraining}
            className="w-full btn-primary disabled:opacity-50"
          >
            {isTraining ? 'Training...' : 'Start Training'}
          </button>

          {message && (
            <div className="mt-4 p-3 rounded bg-blue-900/30 text-blue-300 text-sm">
              {message}
            </div>
          )}
        </div>

        {/* Progress & Charts */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-2">
          <h2 className="text-xl font-semibold mb-4 text-blue-400">Execution Status</h2>
          
          {progress && (
            <div className="mb-6">
              <div className="flex justify-between text-sm mb-2">
                <span>Status: <strong className="text-blue-300 capitalize">{progress.status}</strong></span>
                <span>Epoch: <strong>{progress.epoch} / {progress.num_epochs}</strong></span>
              </div>
              <div className="w-full bg-slate-700 h-2 rounded-full overflow-hidden">
                <div 
                  className="bg-blue-500 h-full transition-all duration-300"
                  style={{ width: `${(progress.epoch / (progress.num_epochs || 1)) * 100}%` }}
                />
              </div>
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="bg-slate-700/30 p-3 rounded text-center">
                  <div className="text-xs text-slate-500 uppercase">Train Loss</div>
                  <div className="text-xl font-bold text-slate-300">{progress.train_loss.toFixed(4)}</div>
                </div>
                <div className="bg-slate-700/30 p-3 rounded text-center">
                  <div className="text-xs text-slate-500 uppercase">Validation Loss</div>
                  <div className="text-xl font-bold text-slate-300">{progress.val_loss.toFixed(4)}</div>
                </div>
              </div>
            </div>
          )}

          {progress && progress.history && progress.history.length > 0 ? (
            <div className="h-64">
              <h3 className="text-sm font-medium mb-2 text-slate-400">Loss Curve</h3>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={progress.history}>
                  <CartesianGrid stroke="#374151" />
                  <XAxis stroke="#6b7280" dataKey="epoch" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                  <Legend />
                  <Line type="monotone" dataKey="train_loss" stroke="#3b82f6" name="Train Loss" dot={false} />
                  <Line type="monotone" dataKey="val_loss" stroke="#f59e0b" name="Val Loss" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center border border-slate-700 rounded bg-slate-900/30 text-slate-500">
              Loss charts will render once training starts.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
