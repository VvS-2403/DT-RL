'use client'

import { usePathname } from 'next/navigation'
import React from 'react'

export default function WorkspaceLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()

  const tabs = [
    { name: 'Dashboard', path: '/' },
    { name: 'Data', path: '/data' },
    { name: 'Train', path: '/train' },
    { name: 'Models', path: '/models' },
    { name: 'Predictions', path: '/predictions' },
    { name: 'Backtest', path: '/backtest' },
    { name: 'Regimes', path: '/regimes' },
    { name: 'Dependencies', path: '/dependencies' },
    { name: 'Settings', path: '/settings' },
  ]

  return (
    <div className="flex min-h-screen">
      {/* Sidebar (Explorer / AI Chat style sidebar) */}
      <nav className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col">
        <div className="p-6 border-b border-gray-700">
          <h1 className="text-xl font-bold text-blue-400">DependencyAI</h1>
          <p className="text-xs text-slate-400 mt-2">Trading System v0.1</p>
        </div>
        
        <div className="p-4 text-xs font-semibold uppercase tracking-wider text-slate-500">
          Navigation
        </div>
        <ul className="space-y-1 px-3 flex-1">
          {tabs.map((tab) => {
            const isActive = pathname === tab.path
            return (
              <li key={tab.path}>
                <a
                  href={tab.path}
                  className={`block px-4 py-2 rounded text-sm transition ${
                    isActive
                      ? 'bg-gray-900 text-white font-medium border-l-2 border-blue-500'
                      : 'text-slate-400 hover:bg-gray-700 hover:text-slate-200'
                  }`}
                >
                  {tab.name}
                </a>
              </li>
            )
          })}
        </ul>
      </nav>
      
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col bg-gray-900">
        {/* VS-Code style Tab Header Bar */}
        <div className="bg-gray-800 border-b border-gray-700 flex items-end h-11 px-2 select-none">
          {tabs.map((tab) => {
            const isActive = pathname === tab.path
            return (
              <a
                key={tab.path}
                href={tab.path}
                className={`h-full flex items-center px-4 text-xs border-r border-gray-700 transition-all ${
                  isActive
                    ? 'bg-gray-900 text-white font-medium border-t-2 border-t-blue-500'
                    : 'bg-gray-800 text-slate-400 hover:bg-gray-700 hover:text-slate-200'
                }`}
              >
                <span>{tab.name}</span>
              </a>
            )
          })}
        </div>

        {/* Editor Workspace Content */}
        <main className="flex-1 overflow-auto bg-gray-900">
          {children}
        </main>
      </div>
    </div>
  )
}
