'use client'

import { useState, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function ModelsPage() {
  const [info, setInfo] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchSystemInfo()
  }, [])

  const fetchSystemInfo = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/health`)
      if (res.ok) {
        const health = await res.ok ? await res.json() : null
        const sysRes = await fetch(`${API_URL}/system-info`)
        const sysInfo = sysRes.ok ? await sysRes.json() : null
        setInfo({ ...health, ...sysInfo })
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Models & Checkpoints</h1>
        <p className="text-slate-400">Manage pipeline configurations, model checkpoints, and PyTorch devices</p>
      </div>

      {loading ? (
        <div className="text-slate-500 py-8 text-center">Loading model settings...</div>
      ) : info ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Active Model Status */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-blue-400">Active Model Status</h2>
            <div className="space-y-4">
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Pipeline Loaded</span>
                <span className="font-semibold">{info.pipeline_loaded ? '✓ Active' : 'No Model Loaded'}</span>
              </div>
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">PyTorch Version</span>
                <span className="font-semibold">v{info.pytorch_version || '2.0.0'}</span>
              </div>
              <div className="flex justify-between border-b border-slate-700 pb-2">
                <span className="text-slate-400">Device Backing</span>
                <span className="font-semibold">{info.gpu_available ? 'NVIDIA GPU (CUDA)' : 'CPU'}</span>
              </div>
              {info.gpu_available && info.gpu_name && (
                <div className="flex justify-between border-b border-slate-700 pb-2">
                  <span className="text-slate-400">GPU Hardware</span>
                  <span className="font-semibold text-sm">{info.gpu_name}</span>
                </div>
              )}
            </div>
          </div>

          {/* Model Checkpoint List */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-blue-400">Model Artifacts</h2>
            <div className="p-4 bg-slate-900/30 rounded border border-slate-700/50 mb-4">
              <div className="text-sm font-semibold text-slate-300">checkpoints/latest_model.pt</div>
              <div className="text-xs text-slate-500 mt-1">Saved automatically upon training completion</div>
            </div>
            <div className="p-4 bg-slate-900/30 rounded border border-slate-700/50">
              <div className="text-sm font-semibold text-slate-300">checkpoints/demo_model.pt</div>
              <div className="text-xs text-slate-500 mt-1">Synthetic data end-to-end model checkpoint</div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-slate-500 py-8 text-center">Failed to fetch server information. Make sure the FastAPI server is running.</div>
      )}
    </div>
  )
}
