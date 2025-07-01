import React from 'react'

function App() {
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full">
        <h1 className="text-2xl font-bold text-center mb-6 text-gray-800">
          CCTV Detection System
        </h1>
        <div className="text-center text-gray-600">
          <p className="mb-4">
            Web interface for your Python-based CCTV detection system.
          </p>
          <p className="text-sm">
            Your Python files (cctv_final.py, trained models) are ready to use.
          </p>
        </div>
      </div>
    </div>
  )
}

export default App