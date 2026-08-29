import type { Metadata } from 'next'
import './globals.css'

import WorkspaceLayout from './WorkspaceLayout'

export const metadata: Metadata = {
  title: 'Dependency-Aware Trading System',
  description: 'Research-grade ML trading dashboard',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-gray-900 text-slate-50">
        <WorkspaceLayout>{children}</WorkspaceLayout>
      </body>
    </html>
  )
}
