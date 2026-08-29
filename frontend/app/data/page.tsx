'use client'

import { useState, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function DataPage() {
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [stats, setStats] = useState<any>(null)
  const [validationReport, setValidationReport] = useState<any>(null)
  const [preview, setPreview] = useState<any>(null)
  const [message, setMessage] = useState('')

  useEffect(() => {
    fetchDataInfo()
  }, [])

  const fetchDataInfo = async () => {
    try {
      const statsRes = await fetch(`${API_URL}/data/stats`)
      if (statsRes.ok) {
        const statsData = await statsRes.json()
        setStats(statsData)
      }
      
      const valRes = await fetch(`${API_URL}/data/validate`)
      if (valRes.ok) {
        const valData = await valRes.json()
        setValidationReport(valData)
      }

      const prevRes = await fetch(`${API_URL}/data/preview`)
      if (prevRes.ok) {
        const prevData = await prevRes.json()
        setPreview(prevData)
      }
    } catch (err) {
      console.error('Error fetching data info:', err)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setMessage('')
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_URL}/data/upload`, {
        method: 'POST',
        body: formData,
      })
      const result = await res.json()
      if (res.ok) {
        setMessage(`✓ Uploaded successfully: ${result.message}`)
        fetchDataInfo()
      } else {
        setMessage(`✗ Upload failed: ${result.detail || 'Unknown error'}`)
      }
    } catch (err: any) {
      setMessage(`✗ Upload failed: ${err.message}`)
    } finally {
      setUploading(false)
    }
  }

  const handleClear = async () => {
    try {
      const res = await fetch(`${API_URL}/data/clear`, { method: 'POST' })
      if (res.ok) {
        setMessage('Dataset cleared.')
        setStats(null)
        setValidationReport(null)
        setPreview(null)
        setFile(null)
      }
    } catch (err) {
      console.error(err)
    }
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Data Management</h1>
        <p className="text-slate-400">Upload, align, and validate financial time-series datasets</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Upload Panel */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-1">
          <h2 className="text-xl font-semibold mb-4 text-blue-400">Upload Dataset</h2>
          <div className="border-2 border-dashed border-slate-600 rounded-lg p-6 text-center hover:border-blue-400 transition mb-4">
            <input 
              type="file" 
              accept=".csv,.parquet" 
              onChange={handleFileChange} 
              className="hidden" 
              id="file-upload" 
            />
            <label htmlFor="file-upload" className="cursor-pointer block">
              <span className="text-slate-300 block mb-2 font-medium">
                {file ? file.name : 'Select CSV or Parquet'}
              </span>
              <span className="text-xs text-slate-500">OHLCV + Technical indicators schema</span>
            </label>
          </div>

          <div className="flex gap-4">
            <button 
              onClick={handleUpload} 
              disabled={!file || uploading} 
              className="flex-1 btn-primary disabled:opacity-50"
            >
              {uploading ? 'Processing...' : 'Upload & Process'}
            </button>
            <button onClick={handleClear} className="btn-secondary">Clear</button>
          </div>

          {message && (
            <div className={`mt-4 p-3 rounded text-sm font-medium ${message.startsWith('✓') ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
              {message}
            </div>
          )}
        </div>

        {/* Stats Panel */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-1">
          <h2 className="text-xl font-semibold mb-4 text-blue-400">Dataset Statistics</h2>
          {stats ? (
            <div className="space-y-4">
              <div>
                <div className="text-xs text-slate-500 uppercase">Number of Assets</div>
                <div className="text-2xl font-bold">{stats.shape.num_assets}</div>
              </div>
              <div>
                <div className="text-xs text-slate-500 uppercase">Timeline Length</div>
                <div className="text-2xl font-bold">{stats.shape.num_days} trading days</div>
              </div>
              <div>
                <div className="text-xs text-slate-500 uppercase">Feature Dimension</div>
                <div className="text-2xl font-bold">{stats.shape.num_features} fields</div>
              </div>
              <div>
                <div className="text-xs text-slate-500 uppercase">Timeline Bounds</div>
                <div className="text-sm font-medium text-slate-300">
                  {stats.date_range.start.split(' ')[0]} to {stats.date_range.end.split(' ')[0]}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-sm">No dataset loaded. Please upload a file.</div>
          )}
        </div>

        {/* Validation Report */}
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-1">
          <h2 className="text-xl font-semibold mb-4 text-blue-400">Data Validation</h2>
          {validationReport ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2">
                <span className={`text-lg font-bold ${validationReport.valid ? 'text-green-400' : 'text-red-400'}`}>
                  {validationReport.valid ? '✓ Valid for Training' : '✗ Issues Found'}
                </span>
              </div>
              <div>
                <div className="text-xs text-slate-500 uppercase mb-1">Data Quality Score</div>
                <div className="w-full bg-slate-700 h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-blue-500 h-full" 
                    style={{ width: `${Math.max(0, validationReport.quality_score * 100)}%` }}
                  />
                </div>
                <div className="text-right text-xs mt-1 text-slate-400">
                  {(validationReport.quality_score * 100).toFixed(0)}%
                </div>
              </div>
              <div>
                <div className="text-xs text-slate-500 uppercase mb-2">Details</div>
                {validationReport.issues.length > 0 ? (
                  <ul className="text-xs text-red-400 space-y-1 max-h-24 overflow-y-auto">
                    {validationReport.issues.map((issue: string, idx: number) => (
                      <li key={idx}>• {issue}</li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-green-400 text-xs">✓ Checks passed successfully. No temporal leakage or NaN issues detected.</div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-sm">Upload data to validate.</div>
          )}
        </div>
      </div>

      {/* Dataset Preview */}
      {preview && (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 text-blue-400">Tensor Alignment Preview</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-slate-300">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="py-2 text-slate-400 font-semibold">Date</th>
                  {preview.assets.map((asset: string) => (
                    <th key={asset} className="py-2 text-slate-400 font-semibold">{asset} (Close)</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.dates.map((date: string, dIdx: number) => (
                  <tr key={date} className="border-b border-slate-700 hover:bg-slate-700/50">
                    <td className="py-2 font-medium">{date.split(' ')[0]}</td>
                    {preview.assets.map((asset: string, aIdx: number) => {
                      const val = preview.data[aIdx]?.[dIdx]?.[3]; // Close is index 3 or similar
                      return (
                        <td key={asset} className="py-2">
                          {val !== undefined && val !== null ? val.toFixed(2) : '-'}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
