'use client'

import React, { useEffect, useRef, useState } from 'react'
import { debounce } from 'lodash'

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [prediction, setPrediction] = useState<number | null>(null)
  const [showExplanation, setShowExplanation] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.strokeStyle = 'white'
        ctx.lineWidth = 10
        ctx.lineCap = 'round'
        ctx.lineJoin = 'round'
      }
    }
  }, [])

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    draw(e)
  }

  const stopDrawing = () => {
    setIsDrawing(false)
    const canvas = canvasRef.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx?.beginPath()
    }
  }

  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (ctx && canvas) {
      const rect = canvas.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      ctx.lineTo(x, y)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(x, y)
    }
  }

  const clearCanvas = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (ctx && canvas) {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
    }
    setPrediction(null)
  }

  const getPrediction = debounce(async () => {
    const canvas = canvasRef.current
    if (canvas) {
      const imageData = canvas.toDataURL('image/png')
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      })
      const data = await response.json()
      setPrediction(data.predicted)
    }
  }, 250)

  useEffect(() => {
    getPrediction()
  }, [isDrawing])

  return (
    <main className="min-h-screen flex flex-col items-center justify-center p-8 bg-black text-white font-mono">
      <h1 className="text-6xl font-extrabold mb-12 text-green-500 transform -skew-x-6">DRAW A NUMBER BETWEEN 1-10</h1>
      <div className="mb-8 text-3xl border-4 border-yellow-400 p-4 transform skew-x-3">
        Prediction: <span className="text-yellow-400">{prediction !== null ? prediction : 'N/A'}</span>
      </div>
      <div className="relative">
        <div className="absolute inset-0 bg-red-500 transform translate-x-2 translate-y-2"></div>
        <canvas
          ref={canvasRef}
          width={280}
          height={280}
          className="relative border-4 border-white bg-gray-900"
          onMouseDown={startDrawing}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
          onMouseMove={draw}
        />
      </div>
      <button
        onClick={clearCanvas}
        className="mt-12 px-8 py-4 bg-white text-black text-2xl font-bold uppercase tracking-widest hover:bg-red-500 hover:text-white transition-colors duration-300 transform hover:scale-105"
      >
        Clear
      </button>
      <div className="mt-12 text-xl border-4 border-green-400 p-4 transform -skew-x-3">
        Created By <a href="https://zackproser.com" target="_blank" rel="noopener noreferrer" className="text-green-400 hover:text-red-500 transition-colors duration-300">Zachary Proser</a>
      </div>
      <button
        onClick={() => setShowExplanation(!showExplanation)}
        className="mt-6 px-6 py-3 bg-green-400 text-black text-xl font-bold uppercase tracking-wider hover:bg-red-500 hover:text-white transition-colors duration-300 transform hover:scale-105"
      >
        What is this?
      </button>
      {showExplanation && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 border-4 border-green-400 p-6 max-w-2xl relative">
            <button
              onClick={() => setShowExplanation(false)}
              className="absolute top-2 right-2 text-green-400 hover:text-red-500 text-2xl font-bold"
            >
              ×
            </button>
            <h2 className="text-2xl font-bold mb-4 text-green-400">About This Project</h2>
            <p className="mb-2">I built this project to better understand neural networks. It uses PyTorch and the MNIST dataset, which is a large collection of handwritten digits commonly used for training image processing systems.</p>
            <p className="mb-2">The model was trained to recognize hand-drawn images and then wrapped in a Flask server. The frontend is built with Next.js and deployed to Vercel.</p>
            <p>When a user draws a number, the frontend sends the image data to the Flask backend. The backend processes the image using the trained PyTorch model and returns a prediction. The frontend then displays this prediction to the user in real-time.</p>
          </div>
        </div>
      )}
    </main>
  )
}
