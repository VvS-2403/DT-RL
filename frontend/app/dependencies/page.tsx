'use client'

import { useState, useEffect } from 'react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export default function DependenciesPage() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [selectedSector, setSelectedSector] = useState('')

  useEffect(() => {
    fetchDependencies()
  }, [])

  const fetchDependencies = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/dependencies/`)
      if (res.ok) {
        const result = await res.json()
        setData(result)
        if (result.sector_names && result.sector_names.length > 0) {
          setSelectedSector(result.sector_names[0])
        }
      }
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Get assets belonging to a given sector (membership probability > threshold)
  const getSectorMembers = (sectorName: string) => {
    if (!data) return []
    const sectorIdx = data.sector_names.indexOf(sectorName)
    if (sectorIdx === -1) return []

    const members = []
    const assets = data.asset_ids
    const H = data.incidence_matrix

    for (let i = 0; i < assets.length; i++) {
      const prob = H[i][sectorIdx]
      if (prob > 0.05) { // Threshold for inclusion
        members.push({
          asset: assets[i],
          probability: prob,
        })
      }
    }

    // Sort by highest probability first
    return members.sort((a, b) => b.probability - a.probability)
  }

  const activeMembers = getSectorMembers(selectedSector)

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Cross-Asset Dependencies</h1>
        <p className="text-slate-400">Latent sectors and hypergraph sector memberships learned by the encoder</p>
      </div>

      {loading ? (
        <div className="text-slate-500 py-12 text-center">Querying hypergraph encoder sectors...</div>
      ) : data ? (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sector Selector */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-1 max-h-screen overflow-y-auto">
            <h2 className="text-xl font-semibold mb-4 text-blue-400">Latent Sectors</h2>
            <div className="space-y-2">
              {data.sector_names.map((sector: string) => {
                const count = getSectorMembers(sector).length
                return (
                  <button
                    key={sector}
                    onClick={() => setSelectedSector(sector)}
                    className={`w-full text-left px-4 py-3 rounded transition flex justify-between items-center ${
                      selectedSector === sector ? 'bg-blue-600 text-white' : 'hover:bg-slate-700 text-slate-300'
                    }`}
                  >
                    <span className="font-semibold">{sector}</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-slate-900 text-slate-400">{count} assets</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Members Detail */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 lg:col-span-3">
            <h2 className="text-xl font-semibold mb-6 text-blue-400">
              Sector Grouping & Membership Strengths ({selectedSector})
            </h2>

            {activeMembers.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {activeMembers.map((member: any) => (
                  <div key={member.asset} className="bg-slate-900/50 p-4 border border-slate-750 rounded-lg flex justify-between items-center">
                    <div>
                      <div className="font-bold text-slate-200 text-base">{member.asset}</div>
                      <div className="text-xs text-slate-500 mt-1">Cross-Asset Hypergraph Member</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-semibold text-blue-400">{(member.probability * 100).toFixed(1)}%</div>
                      <div className="text-xs text-slate-500 uppercase mt-0.5">Strength</div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-slate-500 py-12 text-center">No assets found in this sector.</div>
            )}
          </div>
        </div>
      ) : (
        <div className="text-slate-500 py-12 text-center">No hypergraph dependency data available. Please upload data first.</div>
      )}
    </div>
  )
}
